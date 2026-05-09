"""
Domain-specific tool definitions for customer support agent simulations.

This module defines tools that a customer support agent can use during
conversations with customers. Tools are divided into:
- Read tools: Retrieve information without modifying state
- Write tools: Modify orders, process transactions, update balances
"""

# =============================================================================
# READ TOOLS
# =============================================================================

GET_PRODUCT_INFO = {
    "type": "function",
    "function": {
        "name": "get_product_info",
        "description": "Retrieve product details including name, price, description, category, specifications, and availability for a specific item. Use this when a customer asks about a product or when you need to verify product details for returns/exchanges.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The unique identifier (ASIN or SKU) of the product to look up"
                },
                "product_name": {
                    "type": "string",
                    "description": "The name or partial name of the product to search for (used if product_id is not available)"
                },
                "include_specifications": {
                    "type": "boolean",
                    "description": "Whether to include detailed technical specifications in the response",
                    "default": True
                }
            },
            "required": []
        }
    }
}

GET_ORDER_DETAILS = {
    "type": "function",
    "function": {
        "name": "get_order_details",
        "description": "Retrieve details and current status of an existing order. Returns order items, quantities, prices, shipping status, delivery dates, and payment information. Use this to verify order information before processing returns, exchanges, or modifications.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique order identifier (e.g., 'ORDER-123-456789')"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier to verify order ownership"
                },
                "include_tracking": {
                    "type": "boolean",
                    "description": "Whether to include shipping tracking information",
                    "default": True
                },
                "detail_type": {
                    "type": "string",
                    "enum": ["items", "delivery_dates", "promotions", "payment", "status", "full"],
                    "description": "Which aspect of the order to retrieve. 'items': product names, SKUs, quantities, prices. 'delivery_dates': per-item delivery dates. 'promotions': discounts and bundles applied. 'payment': payment method and totals. 'status': shipping and delivery status. 'full': all order information.",
                    "default": "full"
                }
            },
            "required": ["order_id"]
        }
    }
}

CHECK_INVENTORY = {
    "type": "function",
    "function": {
        "name": "check_inventory",
        "description": "Check current inventory and stock levels for a product. Returns availability status, quantity in stock, estimated restock dates if out of stock, and fulfillment options. Use this before processing exchanges or when customers ask about availability.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The unique identifier (ASIN or SKU) of the product"
                },
                "quantity_needed": {
                    "type": "integer",
                    "description": "The quantity the customer wants to purchase or exchange for",
                    "default": 1
                },
                "fulfillment_center": {
                    "type": "string",
                    "description": "Optional specific fulfillment center to check (e.g., 'US-WEST', 'US-EAST')"
                },
                "check_alternatives": {
                    "type": "boolean",
                    "description": "Whether to suggest similar in-stock alternatives if item is unavailable",
                    "default": False
                }
            },
            "required": ["product_id"]
        }
    }
}

GET_PURCHASE_HISTORY = {
    "type": "function",
    "function": {
        "name": "get_purchase_history",
        "description": "Look up a customer's purchase history. Returns past orders, items purchased, dates, and return/exchange history. Use this to establish context for returns, verify purchase dates for warranty claims, or check customer's relationship with the company.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer's unique account identifier"
                },
                "product_id": {
                    "type": "string",
                    "description": "Optional filter to find purchases of a specific product"
                },
                "date_range_start": {
                    "type": "string",
                    "description": "Start date for filtering purchase history (ISO format: YYYY-MM-DD)"
                },
                "date_range_end": {
                    "type": "string",
                    "description": "End date for filtering purchase history (ISO format: YYYY-MM-DD)"
                },
                "include_returns": {
                    "type": "boolean",
                    "description": "Whether to include return/exchange history in the response",
                    "default": True
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of orders to return",
                    "default": 10
                }
            },
            "required": ["customer_id"]
        }
    }
}

GET_POLICY_INFO = {
    "type": "function",
    "function": {
        "name": "get_policy_info",
        "description": "Fetch domain-specific policy information related to returns, exchanges, refunds, warranties, and discounts. Use this to verify policy details before making decisions or to explain policies to customers.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "enum": ["returns", "exchanges", "refunds", "warranties", "shipping", "discounts", "gift_cards", "damaged_items", "all"],
                    "description": "The type of policy information to retrieve"
                },
                "product_category": {
                    "type": "string",
                    "description": "Optional product category to get category-specific policy details (e.g., 'electronics', 'clothing', 'perishables')"
                },
                "query": {
                    "type": "string",
                    "description": "Specific policy question or keyword to search for within the policy documents"
                }
            },
            "required": ["policy_type"]
        }
    }
}

GET_RETURN_FREQUENCY_ASSESSMENT = {
    "type": "function",
    "function": {
        "name": "get_return_frequency_assessment",
        "description": "Retrieve a customer's return frequency metrics and abuse risk level. Returns return_count, return_rate, abuse_risk_level (LOW/MEDIUM/HIGH), and recommended_deduction_percentage (0, 5, or 10). Call this before finalising any refund or return resolution to determine if a return-frequency penalty applies.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer's unique account identifier"
                },
                "lookback_months": {
                    "type": "integer",
                    "description": "Number of past months to evaluate (default 12)",
                    "default": 12
                }
            },
            "required": ["customer_id"]
        }
    }
}


# =============================================================================
# WRITE TOOLS
# =============================================================================

UPDATE_ORDER = {
    "type": "function",
    "function": {
        "name": "update_order",
        "description": "Modify an existing order. Can update shipping address, delivery instructions, item quantities, or cancel specific items. Use this when a customer wants to make changes to an order that hasn't shipped yet.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique order identifier"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier for verification"
                },
                "update_type": {
                    "type": "string",
                    "enum": ["shipping_address", "delivery_instructions", "quantity", "cancel_item", "cancel_order"],
                    "description": "The type of update to perform"
                },
                "new_shipping_address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "description": "New shipping address (required if update_type is 'shipping_address')"
                },
                "delivery_instructions": {
                    "type": "string",
                    "description": "New delivery instructions (required if update_type is 'delivery_instructions')"
                },
                "item_id": {
                    "type": "string",
                    "description": "The specific item within the order to modify (required for quantity/cancel_item updates)"
                },
                "new_quantity": {
                    "type": "integer",
                    "description": "New quantity for the item (required if update_type is 'quantity')"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the update (required for cancellations)"
                }
            },
            "required": ["order_id", "customer_id", "update_type"]
        }
    }
}

PROCESS_RETURN = {
    "type": "function",
    "function": {
        "name": "process_return",
        "description": "Initiate and process a return for one or more items from an order. Creates a return label, updates order status, and initiates the refund process. Use this when a customer wants to return an item.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique order identifier"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier for verification"
                },
                "resolution_type": {
                    "type": "string",
                    "enum": [
                        "RETURN_REFUND_FULL_BANK",
                        "RETURN_REFUND_PARTIAL_BANK",
                        "RETURN_REFUND_GIFT_CARD",
                        "DENY_REFUND",
                        "ESCALATE_HUMAN_AGENT",
                        "REPLACEMENT_EXCHANGE",
                        "USER_ABORT"
                    ],
                    "description": "The resolution outcome: use RETURN_REFUND_FULL_BANK for full refund to original payment, RETURN_REFUND_PARTIAL_BANK for partial refund, RETURN_REFUND_GIFT_CARD for store credit, DENY_REFUND to reject the return, ESCALATE_HUMAN_AGENT to hand off, REPLACEMENT_EXCHANGE for an exchange, USER_ABORT if customer cancels"
                },
                "items_to_return": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "string",
                                "description": "The unique identifier of the item to return"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Number of units to return"
                            },
                            "condition": {
                                "type": "string",
                                "enum": ["unopened", "opened_unused", "opened_used", "defective", "damaged_in_shipping", "wrong_item"],
                                "description": "Current condition of the item"
                            }
                        },
                        "required": ["item_id", "quantity", "condition"]
                    },
                    "description": "List of items to return with their conditions"
                },
                "return_reason": {
                    "type": "string",
                    "enum": [
                        "changed_mind",
                        "item_not_as_described",
                        "defective_or_damaged",
                        "arrived_too_late",
                        "wrong_item_received",
                        "quality_not_as_expected",
                        "found_better_price",
                        "accidental_order",
                        "other"
                    ],
                    "description": "Primary reason for the return"
                },
                "return_reason_details": {
                    "type": "string",
                    "description": "Additional details about the return reason"
                },
                "refund_method": {
                    "type": "string",
                    "enum": ["original_payment", "store_credit", "gift_card", "bank_transfer"],
                    "description": "How the customer wants to receive the refund",
                    "default": "original_payment"
                },
                "return_shipping_method": {
                    "type": "string",
                    "enum": ["drop_off", "pickup", "mail"],
                    "description": "How the customer will return the item"
                }
            },
            "required": ["order_id", "customer_id", "resolution_type", "items_to_return", "return_reason"]
        }
    }
}

PROCESS_EXCHANGE = {
    "type": "function",
    "function": {
        "name": "process_exchange",
        "description": "Exchange one or more items from an order for different items. Handles the return of original items and shipment of replacement items, including any price adjustments. Use this when a customer wants to swap an item for a different size, color, or product.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The original order identifier"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier for verification"
                },
                "items_to_exchange": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_item_id": {
                                "type": "string",
                                "description": "The item ID of the product being returned"
                            },
                            "original_quantity": {
                                "type": "integer",
                                "description": "Number of units being returned"
                            },
                            "new_product_id": {
                                "type": "string",
                                "description": "The product ID of the replacement item"
                            },
                            "new_quantity": {
                                "type": "integer",
                                "description": "Number of replacement units"
                            },
                            "condition": {
                                "type": "string",
                                "enum": ["unopened", "opened_unused", "opened_used", "defective"],
                                "description": "Condition of the item being returned"
                            }
                        },
                        "required": ["original_item_id", "original_quantity", "new_product_id", "new_quantity", "condition"]
                    },
                    "description": "List of items to exchange"
                },
                "exchange_reason": {
                    "type": "string",
                    "enum": [
                        "wrong_size",
                        "wrong_color",
                        "changed_mind",
                        "defective",
                        "damaged",
                        "upgrade",
                        "downgrade",
                        "other"
                    ],
                    "description": "Reason for the exchange"
                },
                "exchange_reason_details": {
                    "type": "string",
                    "description": "Additional details about the exchange reason"
                },
                "price_difference_handling": {
                    "type": "string",
                    "enum": ["charge_customer", "refund_difference", "waive_difference"],
                    "description": "How to handle any price difference between original and new items"
                },
                "shipping_address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "description": "Shipping address for the replacement item (defaults to original order address)"
                }
            },
            "required": ["order_id", "customer_id", "items_to_exchange", "exchange_reason"]
        }
    }
}

ISSUE_REFUND = {
    "type": "function",
    "function": {
        "name": "issue_refund",
        "description": "Issue a refund or store credit to a customer. Can be a full or partial refund. Use this after a return is processed, for order cancellations, or for goodwill gestures.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier associated with the refund"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier"
                },
                "refund_type": {
                    "type": "string",
                    "enum": ["full", "partial", "shipping_only", "goodwill"],
                    "description": "Type of refund to issue"
                },
                "refund_amount": {
                    "type": "number",
                    "description": "Amount to refund (required for partial and goodwill refunds)"
                },
                "currency": {
                    "type": "string",
                    "description": "Currency code for the refund amount",
                    "default": "USD"
                },
                "refund_method": {
                    "type": "string",
                    "enum": ["original_payment", "store_credit", "gift_card", "bank_transfer", "check"],
                    "description": "Method to use for the refund"
                },
                "refund_reason": {
                    "type": "string",
                    "description": "Reason for issuing the refund"
                },
                "items_refunded": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "amount": {"type": "number"}
                        }
                    },
                    "description": "Specific items being refunded (for partial refunds)"
                },
                "include_shipping": {
                    "type": "boolean",
                    "description": "Whether to include shipping costs in the refund",
                    "default": False
                },
                "include_tax": {
                    "type": "boolean",
                    "description": "Whether to include applicable taxes in the refund",
                    "default": True
                },
                "notes": {
                    "type": "string",
                    "description": "Internal notes about the refund for record-keeping"
                }
            },
            "required": ["order_id", "customer_id", "refund_type", "refund_method", "refund_reason"]
        }
    }
}

APPLY_DISCOUNT = {
    "type": "function",
    "function": {
        "name": "apply_discount",
        "description": "Apply a discount code, promotional offer, or gift card to an order. Can also be used to apply courtesy discounts or price adjustments. Use this when a customer has a valid promo code or when offering a goodwill discount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier to apply the discount to"
                },
                "customer_id": {
                    "type": "string",
                    "description": "The customer's account identifier"
                },
                "discount_type": {
                    "type": "string",
                    "enum": ["promo_code", "gift_card", "percentage_off", "fixed_amount", "free_shipping", "buy_one_get_one", "loyalty_reward"],
                    "description": "Type of discount to apply"
                },
                "discount_code": {
                    "type": "string",
                    "description": "The promotional code or gift card number (required for promo_code and gift_card types)"
                },
                "discount_value": {
                    "type": "number",
                    "description": "Discount amount or percentage (required for percentage_off and fixed_amount types)"
                },
                "apply_to": {
                    "type": "string",
                    "enum": ["entire_order", "specific_items", "shipping", "most_expensive_item"],
                    "description": "What the discount should be applied to"
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific item IDs to apply discount to (required if apply_to is 'specific_items')"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for applying the discount (required for manual/courtesy discounts)"
                },
                "override_restrictions": {
                    "type": "boolean",
                    "description": "Whether to override normal discount restrictions (requires supervisor approval)",
                    "default": False
                }
            },
            "required": ["order_id", "customer_id", "discount_type"]
        }
    }
}


# =============================================================================
# TOOL COLLECTIONS
# =============================================================================

READ_TOOLS = [
    GET_PRODUCT_INFO,
    GET_ORDER_DETAILS,
    CHECK_INVENTORY,
    GET_PURCHASE_HISTORY,
    GET_POLICY_INFO,
    GET_RETURN_FREQUENCY_ASSESSMENT,
]

WRITE_TOOLS = [
    UPDATE_ORDER,
    PROCESS_RETURN,
    PROCESS_EXCHANGE,
    ISSUE_REFUND,
    APPLY_DISCOUNT,
]

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS


# =============================================================================
# CUSTOMER TOOLS
# =============================================================================
# Tools available to the customer (user) during the conversation.
# These represent actions a customer can take independently.

CUSTOMER_WITHDRAW_CONVERSATION = {
    "type": "function",
    "function": {
        "name": "withdraw_from_conversation",
        "description": "End the conversation and disconnect from the support session. Use this when the customer decides to leave the conversation, whether due to frustration, satisfaction, or wanting to try a different approach later.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": [
                        "satisfied_resolved",
                        "frustrated_giving_up",
                        "will_try_later",
                        "prefer_different_channel",
                        "no_longer_need_help",
                        "taking_too_long",
                        "other"
                    ],
                    "description": "Reason for withdrawing from the conversation"
                },
                "reason_details": {
                    "type": "string",
                    "description": "Additional details about why the customer is leaving"
                },
                "will_return": {
                    "type": "boolean",
                    "description": "Whether the customer intends to return later to continue",
                    "default": False
                }
            },
            "required": ["reason"]
        }
    }
}

CUSTOMER_VIEW_ORDER_DETAILS = {
    "type": "function",
    "function": {
        "name": "customer_view_order_details",
        "description": "Customer views their own order details from their account dashboard. Use this when the customer wants to check their order information, tracking status, or purchase details during the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier the customer wants to view"
                },
                "view_type": {
                    "type": "string",
                    "enum": ["summary", "full_details", "tracking_only", "items_only", "payment_info"],
                    "description": "What aspect of the order to view",
                    "default": "summary"
                }
            },
            "required": ["order_id"]
        }
    }
}

CUSTOMER_CHECK_ITEM_AVAILABILITY = {
    "type": "function",
    "function": {
        "name": "customer_check_item_availability",
        "description": "Customer checks if a product is available on the website. Use this when the customer wants to verify stock availability before requesting an exchange or making a new purchase decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product identifier (ASIN/SKU) to check"
                },
                "product_name": {
                    "type": "string",
                    "description": "Product name to search for if ID is not known"
                },
                "check_variants": {
                    "type": "boolean",
                    "description": "Whether to check availability of color/size variants",
                    "default": False
                },
                "preferred_size": {
                    "type": "string",
                    "description": "Specific size variant to check"
                },
                "preferred_color": {
                    "type": "string",
                    "description": "Specific color variant to check"
                }
            },
            "required": []
        }
    }
}

CUSTOMER_CONFIRM_RETURNED_ITEMS = {
    "type": "function",
    "function": {
        "name": "customer_confirm_returned_items",
        "description": "Customer confirms or verifies the status of items they have returned. Use this when the customer wants to check if their return has been received, processed, or refunded.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The original order identifier for the return"
                },
                "return_id": {
                    "type": "string",
                    "description": "The return request identifier if known"
                },
                "check_type": {
                    "type": "string",
                    "enum": ["shipping_status", "receipt_confirmation", "refund_status", "all"],
                    "description": "What aspect of the return to check",
                    "default": "all"
                },
                "tracking_number": {
                    "type": "string",
                    "description": "Return shipment tracking number if available"
                }
            },
            "required": ["order_id"]
        }
    }
}

CUSTOMER_INSPECT_PROFILE = {
    "type": "function",
    "function": {
        "name": "customer_inspect_profile",
        "description": "Customer reviews their account profile information. Use this when the customer wants to verify their account details, membership status, saved addresses, or payment methods during the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": [
                        "basic_info",
                        "addresses",
                        "payment_methods",
                        "membership_status",
                        "communication_preferences",
                        "order_history_summary",
                        "all"
                    ],
                    "description": "Which section of the profile to view",
                    "default": "all"
                },
                "verify_for_agent": {
                    "type": "boolean",
                    "description": "Whether the customer is checking this to provide verification to the agent",
                    "default": False
                }
            },
            "required": []
        }
    }
}


# =============================================================================
# CUSTOMER TOOL COLLECTIONS
# =============================================================================

CUSTOMER_TOOLS = [
    CUSTOMER_WITHDRAW_CONVERSATION,
    CUSTOMER_VIEW_ORDER_DETAILS,
    CUSTOMER_CHECK_ITEM_AVAILABILITY,
    CUSTOMER_CONFIRM_RETURNED_ITEMS,
    CUSTOMER_INSPECT_PROFILE,
]


# =============================================================================
# DETERMINISTIC PROCESS_RETURN RESULTS
# Used by Environment._handle_process_return to return consistent results
# for each resolution type without calling the LLM.
# =============================================================================

PROCESS_RETURN_RESULTS = {
    "RETURN_REFUND_FULL_BANK": {
        "status": "success",
        "action": "return_initiated",
        "return_label_url": "https://returns.example.com/label/{order_id}",
        "refund_method": "original_payment_method",
        "estimated_refund_days": 3,
        "message": "Return initiated. Full refund to your original payment method within 3-5 business days of receiving the item.",
    },
    "RETURN_REFUND_PARTIAL_BANK": {
        "status": "success",
        "action": "return_initiated",
        "return_label_url": "https://returns.example.com/label/{order_id}",
        "refund_method": "original_payment_method",
        "refund_type": "partial",
        "estimated_refund_days": 3,
        "message": "Return initiated. Partial refund to your original payment method within 3-5 business days of receiving the item.",
    },
    "RETURN_REFUND_GIFT_CARD": {
        "status": "success",
        "action": "return_initiated",
        "return_label_url": "https://returns.example.com/label/{order_id}",
        "refund_method": "gift_card",
        "estimated_refund_days": 1,
        "message": "Return initiated. Refund issued as store credit/gift card within 1-2 business days of receiving the item.",
    },
    "DENY_REFUND": {
        "status": "denied",
        "action": "return_denied",
        "message": "Return request denied. The item does not meet the return eligibility criteria per current policy.",
    },
    "ESCALATE_HUMAN_AGENT": {
        "status": "escalated",
        "action": "escalated_to_specialist",
        "ticket_id": "ESC-{order_id}",
        "message": "Case escalated to a specialist. Customer will be contacted within 24-48 hours.",
    },
    "REPLACEMENT_EXCHANGE": {
        "status": "success",
        "action": "exchange_initiated",
        "return_label_url": "https://returns.example.com/label/{order_id}",
        "message": "Exchange initiated. Return the original item using the label provided. Replacement ships within 1-2 business days.",
    },
    "USER_ABORT": {
        "status": "cancelled",
        "action": "request_cancelled",
        "message": "Return request cancelled at customer request.",
    },
}


def format_tools_for_prompt() -> str:
    """Format agent tools as a string suitable for inclusion in LLM prompts."""
    lines = ["## Available Tools\n"]

    lines.append("### Read Tools (Information Retrieval)\n")
    for tool in READ_TOOLS:
        name = tool["function"]["name"]
        desc = tool["function"]["description"].split(".")[0]
        lines.append(f"- **{name}**: {desc}")

    lines.append("\n### Write Tools (State Modification)\n")
    for tool in WRITE_TOOLS:
        name = tool["function"]["name"]
        desc = tool["function"]["description"].split(".")[0]
        lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)


def format_customer_tools_for_prompt() -> str:
    """Format customer tools as a string suitable for inclusion in customer LLM prompts."""
    lines = ["## Available Customer Actions\n"]
    lines.append("You have access to the following tools to perform actions during the conversation:\n")

    for tool in CUSTOMER_TOOLS:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"**{name}**")
        lines.append(f"  Description: {desc}")
        if params:
            lines.append("  Parameters:")
            for param_name, param_info in params.items():
                req_marker = " (required)" if param_name in required else ""
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                lines.append(f"    - {param_name}{req_marker}: {param_type} - {param_desc}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Print tool summary when run directly
    print(format_tools_for_prompt())
    print("\n" + "=" * 60 + "\n")
    print(format_customer_tools_for_prompt())
    print(f"\nTotal tools defined: {len(ALL_TOOLS) + len(CUSTOMER_TOOLS)}")
    print(f"  - Agent Read tools: {len(READ_TOOLS)}")
    print(f"  - Agent Write tools: {len(WRITE_TOOLS)}")
    print(f"  - Customer tools: {len(CUSTOMER_TOOLS)}")
