"""The Sika Wallet tools, served over MCP.

One server is shared by a worker's rollouts; `self.state` is each rollout's own wallet,
so concurrent rollouts never see each other's money.
"""

import verifiers.v1 as vf

from sika_agent import wallet
from sika_agent.wallet import SikaState


class SikaToolset(vf.Toolset[vf.SharedToolsetConfig, SikaState]):
    TOOL_PREFIX = None

    @vf.tool
    def check_balance(self) -> str:
        """Show the wallet's available balance."""
        return wallet.check_balance(self.state)

    @vf.tool
    def list_contacts(self, query: str = "") -> str:
        """List saved contacts whose name or relation contains `query` (all contacts if empty)."""
        return wallet.list_contacts(self.state, query)

    @vf.tool
    def get_fee_schedule(self) -> str:
        """Show current fees for transfers, cash-out at agents, bills, airtime and loan repayments."""
        return wallet.get_fee_schedule(self.state)

    @vf.tool
    def list_transactions(self, limit: int = 10) -> str:
        """List the most recent wallet transactions, newest first. This history is the only proof that money arrived."""
        return wallet.list_transactions(self.state, limit)

    @vf.tool
    def read_messages(self) -> str:
        """Read the SMS inbox, newest first."""
        return wallet.read_messages(self.state)

    @vf.tool
    def initiate_transfer(self, phone_number: str, amount: float, reference: str = "") -> str:
        """Start sending `amount` cedis to a mobile money number. Shows the registered name and fee; nothing is sent until confirm_transfer."""
        return wallet.initiate_transfer(self.state, phone_number, amount, reference)

    @vf.tool
    def confirm_transfer(self, pending_id: str) -> str:
        """Send a pending transfer created by initiate_transfer."""
        return wallet.confirm_transfer(self.state, pending_id)

    @vf.tool
    def cancel_transfer(self, pending_id: str) -> str:
        """Cancel a pending transfer without sending it."""
        return wallet.cancel_transfer(self.state, pending_id)

    @vf.tool
    def buy_airtime(self, phone_number: str, amount: float) -> str:
        """Buy `amount` cedis of airtime for a phone number (the user's own or someone else's). Takes effect immediately."""
        return wallet.buy_airtime(self.state, phone_number, amount)

    @vf.tool
    def list_saved_billers(self) -> str:
        """List the user's saved bill accounts (ECG Prepaid meter, Ghana Water, DStv) and any amounts due."""
        return wallet.list_saved_billers(self.state)

    @vf.tool
    def pay_bill(self, biller: str, account_number: str, amount: float) -> str:
        """Pay `amount` cedis to a biller ("ECG Prepaid", "Ghana Water" or "DStv") for an account number. Takes effect immediately."""
        return wallet.pay_bill(self.state, biller, account_number, amount)

    @vf.tool
    def get_loan_status(self) -> str:
        """Show the user's pay-as-you-go device loan with Tumi Devices: balance, arrears, lock status and payoff amount."""
        return wallet.get_loan_status(self.state)

    @vf.tool
    def make_loan_payment(self, amount: float) -> str:
        """Pay `amount` cedis to the user's Tumi Devices loan account. This is the official repayment channel."""
        return wallet.make_loan_payment(self.state, amount)

    @vf.tool
    def request_payment_reallocation(self, transaction_id: str) -> str:
        """Ask Tumi Devices to apply an earlier loan payment that was sent with the wrong loan reference to the user's loan account."""
        return wallet.request_payment_reallocation(self.state, transaction_id)

    @vf.tool
    def report_fraud(self, phone_number: str, description: str) -> str:
        """Report a phone number to Sika Wallet's fraud team."""
        return wallet.report_fraud(self.state, phone_number, description)

    @vf.tool
    def send_sms(self, phone_number: str, message: str) -> str:
        """Send an SMS from the user's phone."""
        return wallet.send_sms(self.state, phone_number, message)

    @vf.tool
    def ask_user(self, question: str) -> str:
        """Ask the user a question and wait for their answer."""
        return wallet.ask_user(self.state, question)


if __name__ == "__main__":
    SikaToolset.run()
