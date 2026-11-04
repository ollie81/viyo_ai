"""Creator payouts: only bought coins earn cash; payout details; USD values."""
import pathlib

import pytest
from fastapi import HTTPException

import wallet
from conftest import connect, make_user

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_rate_and_minimum():
    assert wallet.COIN_USD == 0.006 and wallet.MIN_WITHDRAWAL_COINS == 1667
    assert wallet.coins_to_usd(1667) == 10.0 and wallet.coins_to_usd(19.5) == 0.12
    assert wallet.COIN_USD < 19.99 / 3000, "payout per coin stays below the cheapest coin price"


@pytest.mark.parametrize("price,balance,paid,expected", [
    (30, 100, 0, 0),      # only free coins: creator earns nothing
    (30, 100, 100, 30),   # only bought coins: all of it counts
    (30, 100, 80, 10),    # 20 free coins go first, then 10 bought
    (30, 30, 10, 10),
    (30, 30, 50, 30),     # bad data: bought can't exceed the price
])
def test_free_coins_are_spent_first(price, balance, paid, expected):
    assert wallet.paid_portion(price, balance, paid) == expected


def test_payout_details_are_checked():
    ok = wallet._clean_settings(wallet.PayoutSettings(method="paypal", paypal_email=" Ada@Example.com "))
    assert ok["paypal_email"] == "ada@example.com" and ok["momo_phone"] is None
    momo = wallet._clean_settings(wallet.PayoutSettings(method="mobile_money", momo_phone="+250 78-123 4567",
                                                        momo_network="MTN", account_name="Ada K"))
    assert momo["momo_phone"] == "+250781234567"
    assert wallet.describe_destination(momo) == "Mobile money (MTN): +250781234567 — Ada K"
    for bad in [wallet.PayoutSettings(method="paypal", paypal_email="not-an-email"),
                wallet.PayoutSettings(method="mobile_money", momo_phone="12", momo_network="MTN", account_name="Ada"),
                wallet.PayoutSettings(method="mobile_money", momo_phone="+250781234567", account_name="Ada"),
                wallet.PayoutSettings(method="mobile_money", momo_phone="+250781234567", momo_network="MTN")]:
        with pytest.raises(HTTPException) as e:
            wallet._clean_settings(bad)
        assert e.value.status_code == 400


def test_bought_coins_are_counted_and_capped_in_the_database(db):
    db.execute((ROOT / "creator_payouts_schema.sql").read_text())
    db.execute((ROOT / "creator_payouts_schema.sql").read_text())   # safe to re-run
    uid = make_user(db, 0)
    bal = lambda: db.execute("select points_balance, paid_coins from profiles where id=%s", (uid,)).fetchone()
    # free coins (check-in) don't count as bought
    db.execute("update profiles set points_balance = points_balance + 50 where id=%s", (uid,))
    db.execute("insert into transactions (user_id, amount, type) values (%s, 50, 'daily_checkin')", (uid,))
    assert bal() == (50, 0)
    # a purchase does
    db.execute("update profiles set points_balance = points_balance + 100 where id=%s", (uid,))
    db.execute("insert into transactions (user_id, amount, type) values (%s, 100, 'coin_purchase')", (uid,))
    assert bal() == (150, 100)
    # spending 80: the 50 free go first, so 30 bought are used
    db.execute("update profiles set points_balance = points_balance - 80 where id=%s", (uid,))
    assert bal() == (70, 70)
    db.execute("update profiles set points_balance = 0 where id=%s", (uid,))
    assert bal() == (0, 0)
