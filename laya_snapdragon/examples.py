"""Inputs for verify and bench: short and long states, three languages, all three question types."""

EMAIL = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

GERMAN = ("Seit dem letzten Update stürzt die App beim Öffnen ab. Ich habe sie schon zweimal neu installiert, "
          "aber es hilft nichts. Bitte um schnelle Hilfe, ich brauche sie morgen für eine Präsentation.")
SPANISH = "¿Tienen descuentos para equipos de más de cincuenta personas? Estamos comparando proveedores este mes."

# ~400 tokens: exercises the 512 bucket (or the CPU, if that bucket isn't built)
LONG = " ".join([
    "Incident report, payments service, written for the weekly review.",
    "At 09:12 UTC the card-authorisation queue started to back up after a configuration push that lowered the",
    "connection-pool size for the ledger database from 64 to 8. Authorisations did not fail outright; they",
    "queued, and the median time to authorise rose from 300 milliseconds to 41 seconds. The mobile app treats",
    "anything over 30 seconds as a timeout and shows the customer an error, so from the customer's side the",
    "payment failed, while on our side it was still pending and in most cases later succeeded.",
    "Around 2,300 customers retried, and roughly 600 of them were charged twice.",
    "The push was rolled back at 10:04 UTC and the queue drained by 10:19. Refunds for the duplicate charges",
    "were issued automatically by 14:00, but support received about 450 tickets and the social team counted",
    "120 public complaints, several from business customers threatening to move to a competitor.",
    "Root cause: the pool size is set in two places, and the review only covered one of them.",
    "Actions: a single source of truth for the pool size, an alert on authorisation latency at 5 seconds,",
    "and making the app show 'pending' instead of an error when authorisation is slow.",
    "Open question for leadership: whether to proactively credit affected business customers this week,",
    "and whether the rollback procedure, which took 52 minutes end to end, needs an owner on every shift.",
] * 2)

CASES = [
    ("email (en)", EMAIL, QUESTIONS),
    ("support (de)", GERMAN, QUESTIONS),
    ("sales (es)", SPANISH, QUESTIONS),
    ("incident, long", LONG, {
        "severity": {"type": "score", "instructions": "How severe was this incident for customers?",
                     "criteria": ["negligible", "minor", "major", "critical"]},
        "needs_leadership": {"type": "noul", "instructions": "Does leadership need to make a decision?"},
        "area": {"type": "choice", "instructions": "Which area caused the incident?",
                 "criteria": ["database configuration", "mobile app", "payment provider", "network"]},
    }),
]
