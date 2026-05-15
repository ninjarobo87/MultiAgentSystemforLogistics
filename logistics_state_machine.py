"""
Logistics Multi-Agent State Machine using LangGraph
Handles: BOLs, Customs Declarations, Shipment Orders, Tracking Updates
"""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from datetime import datetime
import json
import operator


# ============================================================================
# STATE DEFINITION
# ============================================================================

class CompilationError(BaseModel):
    """Structured error log for validation failures"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    iteration: int
    error_type: str  # "missing_field", "format_error", "business_rule_violation"
    field_name: str
    expected: str
    received: str
    agent: str
    resolution_hint: str


class ShipmentData(BaseModel):
    """Extracted logistics data structure"""
    tracking_number: str = ""
    origin_address: str = ""
    origin_country: str = ""
    destination_address: str = ""
    destination_country: str = ""
    weight_kg: float = 0.0
    dimensions_cm: str = ""  # LxWxH
    commodity_description: str = ""
    hs_code: str = ""
    declared_value_usd: float = 0.0
    carrier: str = ""
    service_type: str = ""  # "express", "standard", "freight"
    incoterms: str = ""  # "FOB", "CIF", "DDP", etc.
    package_count: int = 0
    special_handling: list[str] = Field(default_factory=list)
    customs_required: bool = False
    hazmat_class: str = ""
    sender_name: str = ""
    receiver_name: str = ""


class LogisticsState(TypedDict):
    """Global state for the logistics multi-agent graph"""
    # Input
    raw_payload: str
    payload_type: str  # "bol", "customs_declaration", "shipment_order", "tracking_update"
    
    # Routing
    route_decision: str  # "extraction" or "validation"
    routing_reasoning: str
    
    # Extracted Data
    extracted_data: dict
    extraction_confidence: float
    
    # Validation
    validation_passed: bool
    validation_errors: list[str]
    
    # Loop Control (CRITICAL: prevents token bleeding)
    loop_count: int
    max_loops: int  # Hardcoded to 3
    
    # Error Compilation Log
    compilation_errors: Annotated[list[dict], operator.add]
    
    # Final Output
    final_status: str  # "success", "failed_max_retries", "error"
    final_output: dict


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ============================================================================
# ROUTER AGENT
# ============================================================================

def router_agent(state: LogisticsState) -> LogisticsState:
    """
    Evaluates incoming logistics payload and determines routing:
    - If raw/unstructured → DataExtractionAgent
    - If structured/pre-extracted → ValidationAgent
    """
    
    system_prompt = """You are a logistics routing agent for a shipping and freight company.
    
    Analyze the incoming payload and determine the correct processing path:
    
    Route to "extraction" if:
    - The payload is raw text (emails, scanned BOL text, unstructured customs forms)
    - The data needs parsing from natural language into structured fields
    - It's a new document that hasn't been processed before
    - The payload contains mixed/unstructured logistics information
    
    Route to "validation" if:
    - The payload is already in structured JSON/dict format
    - The data has been previously extracted and needs verification
    - It's a re-submission after extraction (loop iteration)
    - The payload contains pre-filled shipment fields
    
    Respond in JSON format:
    {
        "decision": "extraction" or "validation",
        "reasoning": "brief explanation",
        "payload_type": "bol" | "customs_declaration" | "shipment_order" | "tracking_update"
    }
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        Payload to evaluate:
        {state['raw_payload']}
        
        Current loop count: {state.get('loop_count', 0)}
        Has extracted data: {bool(state.get('extracted_data'))}
        """)
    ])
    
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Default to extraction if parsing fails
        result = {
            "decision": "extraction",
            "reasoning": "Failed to parse router response, defaulting to extraction",
            "payload_type": "shipment_order"
        }
    
    # If we're in a retry loop with existing data, force validation path
    # unless the error specifically requires re-extraction
    if state.get('loop_count', 0) > 0 and state.get('extracted_data'):
        result["decision"] = "extraction"  # Re-extract with error context
        result["reasoning"] = f"Re-extraction needed (attempt {state.get('loop_count', 0) + 1}/3)"
    
    return {
        **state,
        "route_decision": result["decision"],
        "routing_reasoning": result["reasoning"],
        "payload_type": result.get("payload_type", state.get("payload_type", "shipment_order")),
    }


# ============================================================================
# DATA EXTRACTION AGENT
# ============================================================================

def data_extraction_agent(state: LogisticsState) -> LogisticsState:
    """
    Extracts structured shipment data from raw logistics payloads.
    On retry loops, uses compilation errors to guide re-extraction.
    """
    
    error_context = ""
    if state.get('compilation_errors'):
        recent_errors = state['compilation_errors'][-5:]  # Last 5 errors
        error_context = f"""
        
        PREVIOUS EXTRACTION ERRORS (fix these in this attempt):
        {json.dumps(recent_errors, indent=2)}
        
        Pay special attention to the fields that failed validation.
        Use the resolution_hints to correct your extraction.
        """
    
    system_prompt = f"""You are a logistics data extraction agent for a global freight company.
    
    Extract ALL relevant shipment information from the raw payload into structured format.
    
    Required fields for a valid shipment:
    - tracking_number (format: XX-XXXXXXXXXX or carrier-specific)
    - origin_address & origin_country
    - destination_address & destination_country  
    - weight_kg (must be > 0)
    - commodity_description
    - hs_code (6-10 digit Harmonized System code)
    - declared_value_usd (must be > 0)
    - carrier (e.g., "DHL", "FedEx", "Maersk", "MSC")
    - service_type ("express", "standard", "freight", "ltl", "ftl")
    - incoterms (valid: FOB, CIF, DDP, EXW, FCA, CPT, CIP, DAP, DPU)
    - package_count (must be >= 1)
    - sender_name & receiver_name
    - customs_required (true if international shipment)
    
    If a field cannot be determined, make a reasonable inference based on context
    or mark as "UNKNOWN" - do NOT leave empty.
    
    Also provide a confidence score (0.0 - 1.0) for overall extraction quality.
    {error_context}
    
    Respond in JSON format with "extracted_data" and "confidence" keys.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        Raw Logistics Payload:
        {state['raw_payload']}
        
        Payload Type: {state.get('payload_type', 'unknown')}
        Extraction Attempt: {state.get('loop_count', 0) + 1}
        """)
    ])
    
    try:
        result = json.loads(response.content)
        extracted = result.get("extracted_data", {})
        confidence = result.get("confidence", 0.5)
    except json.JSONDecodeError:
        extracted = {"raw_response": response.content}
        confidence = 0.3
    
    return {
        **state,
        "extracted_data": extracted,
        "extraction_confidence": confidence,
    }


# ============================================================================
# VALIDATION AGENT
# ============================================================================

def validation_agent(state: LogisticsState) -> LogisticsState:
    """
    Validates extracted logistics data against business rules:
    - Required field presence
    - Format validation (HS codes, tracking numbers)
    - Business logic (weight limits, valid incoterms, country restrictions)
    - Customs compliance checks
    """
    
    extracted = state.get('extracted_data', {})
    errors = []
    compilation_errors = []
    current_iteration = state.get('loop_count', 0) + 1
    
    # ---- RULE 1: Required Fields Check ----
    required_fields = [
        'tracking_number', 'origin_country', 'destination_country',
        'weight_kg', 'commodity_description', 'carrier', 'service_type',
        'package_count', 'sender_name', 'receiver_name'
    ]
    
    for field in required_fields:
        value = extracted.get(field, "")
        if not value or value == "UNKNOWN" or value == "":
            errors.append(f"Missing required field: {field}")
            compilation_errors.append(CompilationError(
                iteration=current_iteration,
                error_type="missing_field",
                field_name=field,
                expected="Non-empty value",
                received=str(value),
                agent="ValidationAgent",
                resolution_hint=f"Extract {field} from the payload context. "
                               f"Look for common logistics identifiers."
            ).model_dump())
    
    # ---- RULE 2: Weight Validation ----
    weight = extracted.get('weight_kg', 0)
    if isinstance(weight, (int, float)):
        if weight <= 0:
            errors.append("Weight must be positive")
            compilation_errors.append(CompilationError(
                iteration=current_iteration,
                error_type="business_rule_violation",
                field_name="weight_kg",
                expected="> 0 kg",
                received=str(weight),
                agent="ValidationAgent",
                resolution_hint="Look for weight indicators: kg, lbs (convert to kg), tonnes"
            ).model_dump())
        elif weight > 30000:  # Max container weight ~30 tonnes
            errors.append(f"Weight {weight}kg exceeds max container limit (30000kg)")
            compilation_errors.append(CompilationError(
                iteration=current_iteration,
                error_type="business_rule_violation",
                field_name="weight_kg",
                expected="<= 30000 kg for single container",
                received=str(weight),
                agent="ValidationAgent",
                resolution_hint="Verify unit conversion. Ensure weight is per-package or per-shipment."
            ).model_dump())
    
    # ---- RULE 3: HS Code Format ----
    hs_code = str(extracted.get('hs_code', ''))
    if hs_code and hs_code != "UNKNOWN":
        cleaned_hs = hs_code.replace('.', '').replace(' ', '')
        if not cleaned_hs.isdigit() or len(cleaned_hs) < 6:
            errors.append(f"Invalid HS code format: {hs_code}")
            compilation_errors.append(CompilationError(
                iteration=current_iteration,
                error_type="format_error",
                field_name="hs_code",
                expected="6-10 digit numeric code (e.g., 8471.30.0100)",
                received=hs_code,
                agent="ValidationAgent",
                resolution_hint="HS codes are numeric. Chapter (2) + Heading (2) + Subheading (2+). "
                               "Look for customs tariff numbers in the payload."
            ).model_dump())
    
    # ---- RULE 4: Incoterms Validation ----
    valid_incoterms = ['EXW', 'FCA', 'CPT', 'CIP', 'DAP', 'DPU', 'DDP', 'FAS', 'FOB', 'CFR', 'CIF']
    incoterms = extracted.get('incoterms', '').upper()
    if incoterms and incoterms != "UNKNOWN" and incoterms not in valid_incoterms:
        errors.append(f"Invalid incoterm: {incoterms}")
        compilation_errors.append(CompilationError(
            iteration=current_iteration,
            error_type="format_error",
            field_name="incoterms",
            expected=f"One of: {valid_incoterms}",
            received=incoterms,
            agent="ValidationAgent",
            resolution_hint="Check for trade terms like FOB, CIF, DDP in the document."
        ).model_dump())
    
    # ---- RULE 5: Customs Requirement Check ----
    origin = extracted.get('origin_country', '').upper()
    destination = extracted.get('destination_country', '').upper()
    customs_required = extracted.get('customs_required', False)
    
    if origin and destination and origin != destination and not customs_required:
        errors.append("International shipment detected but customs_required is False")
        compilation_errors.append(CompilationError(
            iteration=current_iteration,
            error_type="business_rule_violation",
            field_name="customs_required",
            expected="True (international shipment)",
            received="False",
            agent="ValidationAgent",
            resolution_hint=f"Origin ({origin}) != Destination ({destination}), "
                           f"so customs declaration is mandatory."
        ).model_dump())
    
    # ---- RULE 6: Declared Value for International ----
    if customs_required or (origin and destination and origin != destination):
        declared_value = extracted.get('declared_value_usd', 0)
        if not declared_value or declared_value <= 0:
            errors.append("International shipments require declared_value_usd > 0")
            compilation_errors.append(CompilationError(
                iteration=current_iteration,
                error_type="business_rule_violation",
                field_name="declared_value_usd",
                expected="> 0 USD for customs declaration",
                received=str(declared_value),
                agent="ValidationAgent",
                resolution_hint="Look for invoice value, commercial value, or goods value in payload."
            ).model_dump())
    
    # ---- RULE 7: Service Type Validation ----
    valid_services = ['express', 'standard', 'freight', 'ltl', 'ftl', 'air', 'ocean', 'rail', 'ground']
    service = extracted.get('service_type', '').lower()
    if service and service != "unknown" and service not in valid_services:
        errors.append(f"Invalid service_type: {service}")
        compilation_errors.append(CompilationError(
            iteration=current_iteration,
            error_type="format_error",
            field_name="service_type",
            expected=f"One of: {valid_services}",
            received=service,
            agent="ValidationAgent",
            resolution_hint="Determine shipping mode from context: air/ocean/ground and speed: express/standard."
        ).model_dump())
    
    # ---- Determine Pass/Fail ----
    validation_passed = len(errors) == 0
    
    # Use LLM for semantic validation (catches edge cases rules miss)
    if validation_passed:
        semantic_check = llm.invoke([
            SystemMessage(content="""You are a logistics compliance validator. 
            Check if this shipment data is logically consistent:
            - Does the carrier match the service type? (e.g., Maersk = ocean freight)
            - Are dimensions reasonable for the weight?
            - Is the HS code plausible for the commodity?
            
            Respond with JSON: {"passed": true/false, "issues": ["issue1", ...]}"""),
            HumanMessage(content=json.dumps(extracted, indent=2))
        ])
        
        try:
            semantic_result = json.loads(semantic_check.content)
            if not semantic_result.get("passed", True):
                validation_passed = False
                for issue in semantic_result.get("issues", []):
                    errors.append(f"Semantic: {issue}")
                    compilation_errors.append(CompilationError(
                        iteration=current_iteration,
                        error_type="business_rule_violation",
                        field_name="semantic_check",
                        expected="Logically consistent data",
                        received=issue,
                        agent="ValidationAgent",
                        resolution_hint=f"Fix logical inconsistency: {issue}"
                    ).model_dump())
        except json.JSONDecodeError:
            pass  # If semantic check fails to parse, don't block
    
    return {
        **state,
        "validation_passed": validation_passed,
        "validation_errors": errors,
        "compilation_errors": compilation_errors,
        "loop_count": current_iteration,
    }


# ============================================================================
# TERMINAL NODES
# ============================================================================

def success_handler(state: LogisticsState) -> LogisticsState:
    """Handles successful validation - prepares final output"""
    return {
        **state,
        "final_status": "success",
        "final_output": {
            "status": "SHIPMENT_VALIDATED",
            "data": state.get('extracted_data', {}),
            "confidence": state.get('extraction_confidence', 0.0),
            "iterations_used": state.get('loop_count', 1),
            "timestamp": datetime.now().isoformat(),
            "ready_for_dispatch": True,
        }
    }


def failure_handler(state: LogisticsState) -> LogisticsState:
    """Handles max retry exceeded - logs failure with full error trail"""
    return {
        **state,
        "final_status": "failed_max_retries",
        "final_output": {
            "status": "VALIDATION_FAILED_MAX_RETRIES",
            "data": state.get('extracted_data', {}),
            "total_attempts": state.get('loop_count', 0),
            "all_errors": state.get('compilation_errors', []),
            "last_validation_errors": state.get('validation_errors', []),
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": True,
            "escalation_queue": "logistics_ops_manual_review",
        }
    }


# ============================================================================
# CONDITIONAL EDGES (ROUTING LOGIC)
# ============================================================================

def route_after_router(state: LogisticsState) -> str:
    """Determines next node after RouterAgent"""
    decision = state.get('route_decision', 'extraction')
    if decision == "validation" and state.get('extracted_data'):
        return "validation_agent"
    return "data_extraction_agent"


def route_after_validation(state: LogisticsState) -> str:
    """
    Determines next node after ValidationAgent.
    CRITICAL: Enforces max loop count of 3 to prevent token bleeding.
    """
    if state.get('validation_passed', False):
        return "success_handler"
    
    # TOKEN BLEEDING PREVENTION: Hard cap at 3 loops
    current_loop = state.get('loop_count', 0)
    max_loops = state.get('max_loops', 3)
    
    if current_loop >= max_loops:
        return "failure_handler"
    
    # Cycle back to extraction with error context
    return "data_extraction_agent"


def route_after_extraction(state: LogisticsState) -> str:
    """After extraction, always proceed to validation"""
    return "validation_agent"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_logistics_graph() -> StateGraph:
    """Constructs the LangGraph state machine for logistics processing"""
    
    # Initialize the graph with our state schema
    workflow = StateGraph(LogisticsState)
    
    # ---- Add Nodes ----
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("data_extraction_agent", data_extraction_agent)
    workflow.add_node("validation_agent", validation_agent)
    workflow.add_node("success_handler", success_handler)
    workflow.add_node("failure_handler", failure_handler)
    
    # ---- Set Entry Point ----
    workflow.set_entry_point("router_agent")
    
    # ---- Add Conditional Edges ----
    
    # Router → Extraction OR Validation
    workflow.add_conditional_edges(
        "router_agent",
        route_after_router,
        {
            "data_extraction_agent": "data_extraction_agent",
            "validation_agent": "validation_agent",
        }
    )
    
    # Extraction → Validation (always)
    workflow.add_conditional_edges(
        "data_extraction_agent",
        route_after_extraction,
        {
            "validation_agent": "validation_agent",
        }
    )
    
    # Validation → Success OR Retry Extraction OR Failure
    workflow.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "success_handler": "success_handler",
            "data_extraction_agent": "data_extraction_agent",
            "failure_handler": "failure_handler",
        }
    )
    
    # ---- Terminal Edges ----
    workflow.add_edge("success_handler", END)
    workflow.add_edge("failure_handler", END)
    
    # ---- Compile ----
    app = workflow.compile()
    
    return app


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

def process_logistics_payload(raw_payload: str) -> dict:
    """
    Main entry point for processing logistics documents.
    
    Args:
        raw_payload: Raw text/JSON of logistics document
        
    Returns:
        Final processed state with shipment data or error report
    """
    
    # Build the graph
    app = build_logistics_graph()
    
    # Initialize state
    initial_state: LogisticsState = {
        "raw_payload": raw_payload,
        "payload_type": "",
        "route_decision": "",
        "routing_reasoning": "",
        "extracted_data": {},
        "extraction_confidence": 0.0,
        "validation_passed": False,
        "validation_errors": [],
        "loop_count": 0,
        "max_loops": 3,  # HARD CAP - prevents token bleeding
        "compilation_errors": [],
        "final_status": "",
        "final_output": {},
    }
    
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    return final_state


# ============================================================================
# EXAMPLE USAGE - LOGISTICS SCENARIOS
# ============================================================================

if __name__ == "__main__":
    
    # ---- Scenario 1: Raw Bill of Lading ----
    bol_payload = """
    BILL OF LADING
    B/L No: MAEU-SH2024-789456
    
    Shipper: Shanghai Electronics Manufacturing Co., Ltd.
    Address: 1288 Pudong Avenue, Shanghai, China 200120
    
    Consignee: TechDistro Americas Inc.
    Address: 4521 Harbor Blvd, Long Beach, CA 90802, USA
    
    Vessel: MSC GULSUN  |  Voyage: FE945W
    Port of Loading: Shanghai (CNSHA)
    Port of Discharge: Long Beach (USLGB)
    
    Cargo Description:
    - 3x 40ft containers
    - Electronic Components (Printed Circuit Boards)
    - HS Code: 8534.00.0040
    - Total Weight: 18,500 KG (Gross)
    - Total Packages: 450 cartons
    
    Declared Value: USD 2,340,000.00
    Terms: FOB Shanghai
    
    Special Instructions: FRAGILE - Handle with care. Temperature controlled (15-25°C)
    
    Carrier: Mediterranean Shipping Company (MSC)
    Service: Ocean Freight - Standard
    
    Date: 2024-11-15
    """
    
    print("=" * 80)
    print("PROCESSING: Bill of Lading")
    print("=" * 80)
    
    result = process_logistics_payload(bol_payload)
    
    print(f"\nFinal Status: {result['final_status']}")
    print(f"Iterations Used: {result['loop_count']}")
    print(f"Validation Passed: {result['validation_passed']}")
    print(f"\nExtracted Data:")
    print(json.dumps(result.get('extracted_data', {}), indent=2))
    
    if result.get('compilation_errors'):
        print(f"\nCompilation Errors ({len(result['compilation_errors'])} total):")
        for err in result['compilation_errors']:
            print(f"  [{err['iteration']}] {err['error_type']}: {err['field_name']} - {err['resolution_hint']}")
    
    print(f"\nFinal Output:")
    print(json.dumps(result.get('final_output', {}), indent=2))
    
    
    # ---- Scenario 2: Messy Email with Shipment Request ----
    email_payload = """
    From: john.martinez@acmecorp.com
    Subject: URGENT - Need to ship parts to our Munich facility ASAP
    
    Hey logistics team,
    
    We need to get 12 pallets of automotive brake assemblies over to our 
    Munich plant by end of next week. Each pallet weighs about 800kg so 
    total should be around 9.6 tonnes. 
    
    Ship from our warehouse at 2100 Industrial Pkwy, Elkhart, IN 46516.
    Deliver to: Acme Automotive GmbH, Ingolstädter Str. 45, 80807 München, Germany
    
    These are Class 9 miscellaneous dangerous goods (lithium batteries embedded 
    in the brake sensors). Need proper DG documentation.
    
    Value is approx $450K. Use DHL or Kühne+Nagel, whatever is fastest.
    Prefer DDP terms so our German team doesn't deal with customs.
    
    Our PO reference is PO-2024-AC-8834
    
    Thanks,
    John
    """
    
    print("\n" + "=" * 80)
    print("PROCESSING: Unstructured Email Shipment Request")
    print("=" * 80)
    
    result2 = process_logistics_payload(email_payload)
    
    print(f"\nFinal Status: {result2['final_status']}")
    print(f"Iterations Used: {result2['loop_count']}")
    print(f"\nFinal Output:")
    print(json.dumps(result2.get('final_output', {}), indent=2))
