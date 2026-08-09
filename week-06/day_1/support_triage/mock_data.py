"""Mock data standing in for real integrations (Zendesk, Stripe, a real KB/CRM).

Everything here is fabricated for demo purposes. Swapping this module out for
real API calls (Zendesk tickets, Stripe customers, a real knowledge base) is
the natural next step once the graph logic is proven out.
"""

from __future__ import annotations

from typing import TypedDict


class Customer(TypedDict):
    name: str
    plan: str
    mrr_usd: int
    account_age_days: int
    billing_status: str  # "current" | "past_due" | "canceled"
    open_invoices: int


CUSTOMERS: dict[str, Customer] = {
    "cust_101": {
        "name": "Priya Shah",
        "plan": "Pro",
        "mrr_usd": 99,
        "account_age_days": 412,
        "billing_status": "past_due",
        "open_invoices": 1,
    },
    "cust_102": {
        "name": "Marcus Webb",
        "plan": "Free",
        "mrr_usd": 0,
        "account_age_days": 14,
        "billing_status": "current",
        "open_invoices": 0,
    },
    "cust_103": {
        "name": "Elena Vasquez",
        "plan": "Enterprise",
        "mrr_usd": 1499,
        "account_age_days": 980,
        "billing_status": "current",
        "open_invoices": 0,
    },
    "cust_104": {
        "name": "Tom O'Brien",
        "plan": "Pro",
        "mrr_usd": 99,
        "account_age_days": 6,
        "billing_status": "current",
        "open_invoices": 0,
    },
    "cust_105": {
        "name": "Aiko Tanaka",
        "plan": "Free",
        "mrr_usd": 0,
        "account_age_days": 2,
        "billing_status": "current",
        "open_invoices": 0,
    },
}


class KBArticle(TypedDict):
    title: str
    keywords: list[str]
    content: str


KNOWLEDGE_BASE: list[KBArticle] = [
    {
        "title": "Resetting API keys",
        "keywords": ["api key", "token", "401", "unauthorized", "authentication"],
        "content": (
            "API keys can be rotated from Settings > Developer > API Keys. "
            "Old keys remain valid for 24 hours after rotation to allow for "
            "a graceful transition. A 401 response almost always means the "
            "key used is expired, revoked, or was copied with trailing "
            "whitespace."
        ),
    },
    {
        "title": "Webhook delivery failures",
        "keywords": ["webhook", "not receiving", "delivery failed", "timeout"],
        "content": (
            "Webhook endpoints must respond with a 2xx status within 5 "
            "seconds or the delivery is marked failed and retried with "
            "exponential backoff (up to 5 attempts over 24 hours). Check "
            "the Webhook Logs page for the exact failure reason per "
            "attempt."
        ),
    },
    {
        "title": "Data export taking too long / stuck",
        "keywords": ["export", "stuck", "slow", "csv", "download"],
        "content": (
            "Exports over 500k rows are processed asynchronously and "
            "emailed as a download link when ready (typically 5-30 "
            "minutes). If an export has been 'processing' for over an "
            "hour, it likely failed silently — the fix is to re-trigger it "
            "with a narrower date range."
        ),
    },
    {
        "title": "SSO / SAML login errors",
        "keywords": ["sso", "saml", "login", "sign in", "idp"],
        "content": (
            "Most SSO login failures trace back to a clock skew between "
            "the identity provider and our servers (SAML assertions are "
            "time-bound), or an IdP metadata URL that changed after a "
            "provider-side certificate rotation."
        ),
    },
]


class PlanInfo(TypedDict):
    name: str
    price_usd_per_month: int
    seats_included: int
    highlights: list[str]


PLANS: list[PlanInfo] = [
    {
        "name": "Free",
        "price_usd_per_month": 0,
        "seats_included": 1,
        "highlights": ["Core features", "Community support", "1,000 API calls/mo"],
    },
    {
        "name": "Pro",
        "price_usd_per_month": 99,
        "seats_included": 5,
        "highlights": [
            "Everything in Free",
            "Priority email support",
            "100,000 API calls/mo",
            "Webhooks",
        ],
    },
    {
        "name": "Enterprise",
        "price_usd_per_month": 1499,
        "seats_included": 50,
        "highlights": [
            "Everything in Pro",
            "Dedicated Slack channel",
            "SSO/SAML",
            "Custom API rate limits",
            "99.9% uptime SLA",
        ],
    },
]


class MockTicket(TypedDict):
    ticket_id: str
    customer_id: str
    subject: str
    message: str


MOCK_TICKETS: list[MockTicket] = [
    {
        "ticket_id": "TCK-1001",
        "customer_id": "cust_101",
        "subject": "Card declined but I was charged twice",
        "message": (
            "Hi, my card shows two charges of $99 from you this month but "
            "the app says my account is past due and I'm locked out. Can "
            "you refund the duplicate and unlock my account? This is "
            "urgent, I have a client demo in an hour."
        ),
    },
    {
        "ticket_id": "TCK-1002",
        "customer_id": "cust_102",
        "subject": "Getting 401 errors from the API all of a sudden",
        "message": (
            "My integration was working fine yesterday and today every "
            "request returns 401 Unauthorized. I didn't change anything "
            "on my end. Here's my key prefix: sk_live_9f2... What's going "
            "on?"
        ),
    },
    {
        "ticket_id": "TCK-1003",
        "customer_id": "cust_103",
        "subject": "Interested in adding 20 more seats + SSO for a subsidiary",
        "message": (
            "We're spinning up a new subsidiary and want to bring them "
            "onto our existing Enterprise plan with their own SSO config. "
            "Can someone walk us through pricing for +20 seats and confirm "
            "SSO can be scoped per business unit?"
        ),
    },
    {
        "ticket_id": "TCK-1004",
        "customer_id": "cust_104",
        "subject": "Webhooks stopped arriving since last night",
        "message": (
            "We haven't received a single webhook event since ~11pm PT "
            "last night. Our endpoint is healthy (we checked uptime), and "
            "nothing changed in our config. Are you having an outage?"
        ),
    },
    {
        "ticket_id": "TCK-1005",
        "customer_id": "cust_105",
        "subject": "Do you offer a student or nonprofit discount?",
        "message": (
            "I'm building a side project for a university club and was "
            "wondering if there's a discount on the Pro plan for students "
            "or nonprofits before I commit."
        ),
    },
    {
        "ticket_id": "TCK-1006",
        "customer_id": "cust_101",
        "subject": "Export has been stuck on 'processing' for 3 hours",
        "message": (
            "I kicked off a full data export at 9am and it's now noon and "
            "still says 'processing'. It's about 2 million rows. Is it "
            "actually running or did it die?"
        ),
    },
]
