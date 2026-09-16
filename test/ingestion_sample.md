# Automated Return System

This document is a sample source for the ingestion pipeline.
It contains representative content for parser and validation tests.

## Return Eligibility

A return request is eligible when the item is unopened and the request is submitted within 30 days.
The original order number must be included in the request.

### Required Information

- Order number
- Customer email address
- Reason for return
- Photos of damaged items, when applicable

## Processing Targets

| Return type | Target processing time | Refund method |
| --- | ---: | --- |
| Unopened item | 2 business days | Original payment method |
| Damaged item | 5 business days | Original payment method |
| Wrong item shipped | 3 business days | Original payment method |

## Example Payload

```json
{
  "order_id": "ORD-2026-0001",
  "reason": "damaged",
  "requested_action": "refund"
}
```

## Notes

Keep the tracking number after the parcel is handed to the carrier.
The support team should record every status change for audit purposes.
