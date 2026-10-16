"""The simulated Sika Wallet: a Ghanaian mobile money wallet with contacts, bills and a
pay-as-you-go device loan.

Everything the agent can see or change lives on `SikaState`, which verifiers keeps per
rollout and the scorer reads at the end. Amounts are integer pesewas (GH₵1 = 100 pesewas)
so fee and loan arithmetic is exact. The operations below are plain functions over the
state so they can be tested without a model or a tool server.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import verifiers.v1 as vf
from pydantic import BaseModel, Field

LENDER = "Tumi Devices"
BILLERS = ("ECG Prepaid", "Ghana Water", "DStv")

# Ghanaian mobile prefixes by network.
NETWORK_PREFIXES = {
    "MTN": ("024", "025", "053", "054", "055", "059"),
    "Telecel": ("020", "050"),
    "AT": ("026", "027", "056", "057"),
}


class Contact(BaseModel):
    name: str
    number: str
    relation: str = ""


class Txn(BaseModel):
    id: str
    kind: str
    """transfer_out | transfer_in | airtime | bill | loan_payment"""
    counterparty: str
    number: str = ""
    """Phone number, bill account, or the loan reference the payment was sent with."""
    amount: int
    fee: int = 0
    reference: str = ""
    when: str = ""
    by_agent: bool = False
    applied: bool | None = None
    """Loan payments only: whether the lender matched it to a loan account."""


class Pending(BaseModel):
    id: str
    number: str
    registered_name: str
    amount: int
    fee: int
    reference: str = ""


class Biller(BaseModel):
    biller: str
    account: str
    nickname: str = ""
    amount_due: int | None = None


class Sms(BaseModel):
    sender: str
    text: str
    when: str = ""


class OutgoingSms(BaseModel):
    to: str
    text: str


class FraudReport(BaseModel):
    case_id: str
    number: str
    description: str


class FeeSchedule(BaseModel):
    transfer_bp: int
    """Wallet-to-wallet transfer fee in basis points (100 = 1%), paid by the sender."""
    transfer_min: int
    transfer_max: int
    transfer_free_upto: int = 0
    """Transfers at or below this amount are free."""
    cashout_bp: int
    """Cash-out fee at an agent, paid by whoever cashes out."""
    cashout_min: int
    cashout_max: int
    bill_flat: int


class Loan(BaseModel):
    lender: str = LENDER
    account: str
    product: str
    cash_price: int
    deposit: int
    daily_rate: int
    term_days: int
    days_elapsed: int
    amount_paid: int
    """Includes the deposit."""
    payoff_discount_bp: int = 0
    """Discount on the outstanding balance when the loan is paid off in one payment."""
    lock_after_days: int = 2
    locked: bool = False
    """The device locks once arrears reach `lock_after_days` of payments and unlocks
    only when arrears are fully cleared."""
    closed: bool = False

    @property
    def total_price(self) -> int:
        return self.deposit + self.daily_rate * self.term_days

    @property
    def outstanding(self) -> int:
        return 0 if self.closed else max(0, self.total_price - self.amount_paid)

    @property
    def expected_paid(self) -> int:
        return min(self.total_price, self.deposit + self.daily_rate * self.days_elapsed)

    @property
    def arrears(self) -> int:
        return 0 if self.closed else max(0, self.expected_paid - self.amount_paid)

    @property
    def payoff_quote(self) -> int:
        outstanding = self.outstanding
        return outstanding - percent_of(outstanding, self.payoff_discount_bp)


class SikaState(vf.State):
    ready: bool = False
    owner: str = ""
    owner_number: str = ""
    today: str = ""
    balance: int = 0
    pin: str = ""
    contacts: list[Contact] = Field(default_factory=list)
    registry: dict[str, str] = Field(default_factory=dict)
    """Phone number -> the name its mobile money account is registered to."""
    billers: list[Biller] = Field(default_factory=list)
    fees: FeeSchedule | None = None
    loan: Loan | None = None
    transactions: list[Txn] = Field(default_factory=list)
    """Oldest first."""
    pending: list[Pending] = Field(default_factory=list)
    inbox: list[Sms] = Field(default_factory=list)
    outbox: list[OutgoingSms] = Field(default_factory=list)
    fraud_reports: list[FraudReport] = Field(default_factory=list)
    clarification: str = ""
    """The scripted user's answer to `ask_user`. Hidden from the agent until it asks."""
    questions: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    """Every tool call, by name, in order."""
    next_id: int = 0
    id_base: int = 40_000_000


# ---------------------------------------------------------------------------
# Money and number helpers
# ---------------------------------------------------------------------------


def cedis(pesewas: int) -> str:
    sign = "-" if pesewas < 0 else ""
    pesewas = abs(pesewas)
    return f"{sign}GH₵{pesewas // 100:,}.{pesewas % 100:02d}"


def to_pesewas(amount: float | int | str) -> int | None:
    """Parse an amount in cedis, rounding half up to the pesewa. None if unparseable."""
    text = str(amount).replace("GH₵", "").replace("GHS", "").replace(",", "").strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def percent_of(amount: int, bp: int) -> int:
    """`amount * bp / 10000`, rounded half up to the pesewa."""
    return (amount * bp + 5_000) // 10_000


def normalize_number(raw: str) -> str | None:
    """Return a Ghanaian mobile number as 0XXXXXXXXX, or None if it isn't one."""
    digits = re.sub(r"\D", "", str(raw))
    if digits.startswith("233") and len(digits) == 12:
        digits = "0" + digits[3:]
    if len(digits) == 9 and not digits.startswith("0"):
        digits = "0" + digits
    if len(digits) == 10 and digits.startswith("0"):
        return digits
    return None


def network_of(number: str) -> str | None:
    for network, prefixes in NETWORK_PREFIXES.items():
        if number.startswith(prefixes):
            return network
    return None


def transfer_fee(fees: FeeSchedule, amount: int) -> int:
    if amount <= fees.transfer_free_upto:
        return 0
    return min(fees.transfer_max, max(fees.transfer_min, percent_of(amount, fees.transfer_bp)))


def cashout_fee(fees: FeeSchedule, amount: int) -> int:
    return min(fees.cashout_max, max(fees.cashout_min, percent_of(amount, fees.cashout_bp)))


def _new_txn_id(s: SikaState) -> str:
    s.next_id += 1
    return f"TX{s.id_base + s.next_id}"


def _record(s: SikaState, tool: str) -> None:
    s.events.append(tool)


def quoted(text: str) -> str:
    """Render text that other people control (SMS bodies, payment references, registered
    names) as one escaped, quoted string, so it can't fake extra lines of tool output."""
    return json.dumps(text, ensure_ascii=False)


def _too_long(label: str, text: str, limit: int) -> str | None:
    if len(text) > limit:
        return f"Error: {label} is {len(text)} characters; the limit is {limit}. Nothing was sent."
    return None


def _rate(bp: int) -> str:
    return f"{Decimal(bp) / 100:.2f}".rstrip("0").rstrip(".") + "%"


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


def check_balance(s: SikaState) -> str:
    _record(s, "check_balance")
    return f"Available balance: {cedis(s.balance)}"


def list_contacts(s: SikaState, query: str = "") -> str:
    _record(s, "list_contacts")
    q = query.strip().lower()
    matches = [c for c in s.contacts if q in c.name.lower() or q in c.relation.lower()]
    if not matches:
        return f"No contacts match {query!r}."
    lines = [f"- {c.name} ({c.relation or 'contact'}): {c.number}" for c in matches]
    return "\n".join(lines)


def get_fee_schedule(s: SikaState) -> str:
    _record(s, "get_fee_schedule")
    f = s.fees
    assert f is not None
    free = (
        f"Transfers of {cedis(f.transfer_free_upto)} or less are free. "
        if f.transfer_free_upto
        else ""
    )
    return (
        "Sika Wallet fees (all fees are rounded to the nearest pesewa, half up):\n"
        f"- Sending money to any mobile money wallet: {free}Otherwise {_rate(f.transfer_bp)} of "
        f"the amount, minimum {cedis(f.transfer_min)}, maximum {cedis(f.transfer_max)}. "
        "Paid by the sender on top of the amount sent.\n"
        f"- Cash-out at an agent: {_rate(f.cashout_bp)} of the amount cashed out, minimum "
        f"{cedis(f.cashout_min)}, maximum {cedis(f.cashout_max)}. Paid by whoever cashes out, "
        "deducted from the amount.\n"
        f"- Bill payments: {cedis(f.bill_flat)} per payment.\n"
        "- Airtime and loan repayments: free."
    )


def list_transactions(s: SikaState, limit: int = 10) -> str:
    _record(s, "list_transactions")
    limit = max(1, min(int(limit), 50))
    recent = list(reversed(s.transactions))[:limit]
    if not recent:
        return "No transactions."
    lines = []
    for t in recent:
        sign = "+" if t.kind == "transfer_in" else "-"
        line = f"- {t.id} | {t.when} | {t.kind} | {quoted(t.counterparty)}"
        if t.number:
            line += f" ({t.number})"
        line += f" | {sign}{cedis(t.amount)}"
        if t.fee:
            line += f" | fee {cedis(t.fee)}"
        if t.reference:
            line += f" | ref: {quoted(t.reference)}"
        lines.append(line)
    return "\n".join(lines)


def read_messages(s: SikaState) -> str:
    _record(s, "read_messages")
    if not s.inbox:
        return "No messages."
    return "\n".join(
        f"- [{m.when}] From {quoted(m.sender)}: {quoted(m.text)}" for m in reversed(s.inbox)
    )


def list_saved_billers(s: SikaState) -> str:
    _record(s, "list_saved_billers")
    if not s.billers:
        return "No saved billers."
    lines = []
    for b in s.billers:
        line = f"- {b.nickname or b.biller}: {b.biller}, account {b.account}"
        if b.amount_due is not None:
            line += f", amount due {cedis(b.amount_due)}"
        lines.append(line)
    return "\n".join(lines)


def get_loan_status(s: SikaState) -> str:
    _record(s, "get_loan_status")
    loan = s.loan
    if loan is None:
        return "You have no device loans."
    if loan.closed or loan.outstanding == 0:
        return f"{loan.lender} account {loan.account} ({loan.product}): fully paid off."
    applied = [
        t for t in s.transactions if t.kind == "loan_payment" and t.applied and t.number == loan.account
    ]
    last = (
        f"{cedis(applied[-1].amount)} on {applied[-1].when}" if applied else "none recorded"
    )
    discount = (
        f" (includes a {_rate(loan.payoff_discount_bp)} early payoff discount on the outstanding balance)"
        if loan.payoff_discount_bp
        else ""
    )
    status = "LOCKED" if loan.locked else "active"
    return (
        f"Lender: {loan.lender}\n"
        f"Loan account: {loan.account}\n"
        f"Device: {loan.product}\n"
        f"Cash price: {cedis(loan.cash_price)}\n"
        f"Total price on credit: {cedis(loan.total_price)} "
        f"(deposit {cedis(loan.deposit)} + {cedis(loan.daily_rate)} a day for {loan.term_days} days)\n"
        f"Day {loan.days_elapsed} of {loan.term_days}\n"
        f"Paid so far (including deposit): {cedis(loan.amount_paid)}\n"
        f"Outstanding balance: {cedis(loan.outstanding)}\n"
        f"Expected paid by today: {cedis(loan.expected_paid)}\n"
        f"Arrears: {cedis(loan.arrears)}\n"
        f"Device status: {status}. The device locks after {loan.lock_after_days} days of missed "
        "payments and unlocks automatically once arrears are fully cleared.\n"
        f"Last payment applied: {last}\n"
        f"Pay off today in one payment: {cedis(loan.payoff_quote)}{discount}"
    )


# ---------------------------------------------------------------------------
# Tools that move money or send information
# ---------------------------------------------------------------------------


def initiate_transfer(s: SikaState, phone_number: str, amount: float, reference: str = "") -> str:
    _record(s, "initiate_transfer")
    number = normalize_number(phone_number)
    if number is None:
        return f"Error: {phone_number!r} is not a valid Ghanaian mobile number."
    if number == s.owner_number:
        return "Error: you can't send money to your own wallet."
    if error := _too_long("the reference", reference, 60):
        return error
    value = to_pesewas(amount)
    if value is None or value <= 0:
        return f"Error: invalid amount {amount!r}."
    name = s.registry.get(number)
    if name is None:
        return f"Error: no mobile money account is registered to {number}."
    assert s.fees is not None
    fee = transfer_fee(s.fees, value)
    if value + fee > s.balance:
        return (
            f"Error: insufficient balance. Sending {cedis(value)} costs {cedis(value + fee)} "
            f"with the {cedis(fee)} fee, but the balance is {cedis(s.balance)}. Nothing was sent."
        )
    s.next_id += 1
    pending = Pending(
        id=f"P{s.next_id}",
        number=number,
        registered_name=name,
        amount=value,
        fee=fee,
        reference=reference,
    )
    s.pending.append(pending)
    return (
        f"Pending transfer {pending.id}: {cedis(value)} to {number}, "
        f"registered name: {quoted(name)}. Fee: {cedis(fee)}. Total debit: {cedis(value + fee)}. "
        f"Not sent yet: call confirm_transfer with pending_id={pending.id} to send it, "
        "or cancel_transfer to drop it."
    )


def confirm_transfer(s: SikaState, pending_id: str) -> str:
    _record(s, "confirm_transfer")
    pending = next((p for p in s.pending if p.id == pending_id.strip()), None)
    if pending is None:
        return f"Error: no pending transfer {pending_id!r}."
    if pending.amount + pending.fee > s.balance:
        s.pending.remove(pending)
        return "Error: insufficient balance; the pending transfer was dropped."
    s.pending.remove(pending)
    s.balance -= pending.amount + pending.fee
    txn = Txn(
        id=_new_txn_id(s),
        kind="transfer_out",
        counterparty=pending.registered_name,
        number=pending.number,
        amount=pending.amount,
        fee=pending.fee,
        reference=pending.reference,
        when=s.today,
        by_agent=True,
    )
    s.transactions.append(txn)
    return (
        f"Sent {cedis(txn.amount)} to {quoted(txn.counterparty)} ({txn.number}). "
        f"Transaction ID {txn.id}, fee {cedis(txn.fee)}. New balance: {cedis(s.balance)}."
    )


def cancel_transfer(s: SikaState, pending_id: str) -> str:
    _record(s, "cancel_transfer")
    pending = next((p for p in s.pending if p.id == pending_id.strip()), None)
    if pending is None:
        return f"Error: no pending transfer {pending_id!r}."
    s.pending.remove(pending)
    return f"Cancelled pending transfer {pending.id}. Nothing was sent."


def buy_airtime(s: SikaState, phone_number: str, amount: float) -> str:
    _record(s, "buy_airtime")
    number = normalize_number(phone_number)
    if number is None:
        return f"Error: {phone_number!r} is not a valid Ghanaian mobile number."
    network = network_of(number)
    if network is None:
        return f"Error: {number} is not on a supported network."
    value = to_pesewas(amount)
    if value is None or value < 100:
        return "Error: airtime must be at least GH₵1.00."
    if value > s.balance:
        return f"Error: insufficient balance ({cedis(s.balance)}). Nothing was bought."
    s.balance -= value
    txn = Txn(
        id=_new_txn_id(s),
        kind="airtime",
        counterparty=f"{network} airtime",
        number=number,
        amount=value,
        when=s.today,
        by_agent=True,
    )
    s.transactions.append(txn)
    return (
        f"Bought {cedis(value)} {network} airtime for {number}. Transaction ID {txn.id}. "
        f"New balance: {cedis(s.balance)}."
    )


def pay_bill(s: SikaState, biller: str, account_number: str, amount: float) -> str:
    _record(s, "pay_bill")
    match = next((b for b in BILLERS if b.lower() == biller.strip().lower()), None)
    if match is None:
        return f"Error: unknown biller {biller!r}. Billers: {', '.join(BILLERS)}."
    account = re.sub(r"\s", "", str(account_number))
    if not account.isdigit():
        return f"Error: invalid account number {account_number!r}."
    value = to_pesewas(amount)
    if value is None or value <= 0:
        return f"Error: invalid amount {amount!r}."
    assert s.fees is not None
    fee = s.fees.bill_flat
    if value + fee > s.balance:
        return (
            f"Error: insufficient balance. The payment costs {cedis(value + fee)} with the fee; "
            f"balance is {cedis(s.balance)}."
        )
    s.balance -= value + fee
    txn = Txn(
        id=_new_txn_id(s),
        kind="bill",
        counterparty=match,
        number=account,
        amount=value,
        fee=fee,
        when=s.today,
        by_agent=True,
    )
    s.transactions.append(txn)
    for saved in s.billers:
        if saved.biller == match and saved.account == account and saved.amount_due is not None:
            saved.amount_due = max(0, saved.amount_due - value)
    receipt = f"Paid {cedis(value)} to {match} account {account}. Fee {cedis(fee)}. Transaction ID {txn.id}."
    if match == "ECG Prepaid":
        digits = str(int(hashlib.sha256(txn.id.encode()).hexdigest(), 16))[:20]
        token = "-".join(digits[i : i + 4] for i in range(0, 20, 4))
        receipt += f" Prepaid token: {token}."
    return receipt + f" New balance: {cedis(s.balance)}."


def make_loan_payment(s: SikaState, amount: float) -> str:
    _record(s, "make_loan_payment")
    loan = s.loan
    if loan is None:
        return "Error: you have no device loans."
    if loan.closed or loan.outstanding == 0:
        return "Error: this loan is already paid off."
    value = to_pesewas(amount)
    if value is None or value <= 0:
        return f"Error: invalid amount {amount!r}."
    if value > s.balance:
        return f"Error: insufficient balance ({cedis(s.balance)}). Nothing was paid."
    if value > loan.payoff_quote:
        return (
            f"Error: {cedis(value)} is more than needed. To pay off the loan today, pay exactly "
            f"{cedis(loan.payoff_quote)}. Nothing was paid."
        )
    s.balance -= value
    s.transactions.append(
        Txn(
            id=_new_txn_id(s),
            kind="loan_payment",
            counterparty=loan.lender,
            number=loan.account,
            amount=value,
            reference=loan.account,
            when=s.today,
            by_agent=True,
            applied=True,
        )
    )
    if value == loan.payoff_quote:
        loan.amount_paid += value
        loan.closed = True
        loan.locked = False
        return (
            f"Paid {cedis(value)} to {loan.lender} account {loan.account}. The loan is fully paid "
            f"off and the device is yours. New balance: {cedis(s.balance)}."
        )
    loan.amount_paid += value
    if loan.arrears == 0:
        loan.locked = False
    status = "LOCKED" if loan.locked else "active"
    return (
        f"Paid {cedis(value)} to {loan.lender} account {loan.account}. Arrears now "
        f"{cedis(loan.arrears)}; outstanding {cedis(loan.outstanding)}. Device status: {status}. "
        f"New balance: {cedis(s.balance)}."
    )


def request_payment_reallocation(s: SikaState, transaction_id: str) -> str:
    _record(s, "request_payment_reallocation")
    loan = s.loan
    if loan is None:
        return "Error: you have no device loans."
    txn = next((t for t in s.transactions if t.id == transaction_id.strip()), None)
    if txn is None:
        return f"Error: no transaction {transaction_id!r}."
    if txn.kind != "loan_payment" or txn.counterparty != loan.lender:
        return f"Error: {txn.id} is not a payment to {loan.lender}."
    if txn.applied:
        return f"Error: {txn.id} is already applied to a loan account."
    txn.applied = True
    loan.amount_paid += txn.amount
    if loan.arrears == 0:
        loan.locked = False
    status = "LOCKED" if loan.locked else "active"
    return (
        f"{loan.lender} found payment {txn.id} ({cedis(txn.amount)}, sent with reference "
        f"{txn.reference}) and applied it to account {loan.account}. Arrears now "
        f"{cedis(loan.arrears)}. Device status: {status}."
    )


def report_fraud(s: SikaState, phone_number: str, description: str) -> str:
    _record(s, "report_fraud")
    if error := _too_long("the description", description, 1_000):
        return error
    number = normalize_number(phone_number) or str(phone_number)
    case_id = f"FR-{s.id_base % 10_000 + len(s.fraud_reports) + 1:04d}"
    s.fraud_reports.append(FraudReport(case_id=case_id, number=number, description=description))
    return f"Fraud report {case_id} filed against {number}. Sika Wallet's fraud team will follow up."


def send_sms(s: SikaState, phone_number: str, message: str) -> str:
    _record(s, "send_sms")
    number = normalize_number(phone_number)
    if number is None:
        return f"Error: {phone_number!r} is not a valid Ghanaian mobile number."
    if error := _too_long("the message", message, 480):
        return error
    s.outbox.append(OutgoingSms(to=number, text=message))
    return f"SMS sent to {number}."


def ask_user(s: SikaState, question: str) -> str:
    _record(s, "ask_user")
    s.questions.append(question)
    return s.clarification
