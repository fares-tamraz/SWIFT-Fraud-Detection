import pandas as pd     # this is like excel in python, it is used to store and manipulate data
import numpy as np      # this is a math library
import random           # this generates random no. we will use it to generate random transactions

np.random.seed(42)
random.seed(42)

COMMON_COUNTRIES = ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia"]
RISKY_COUNTRIES = ["Nigeria", "Russia", "North Korea", "Iran", "Cayman Islands", "Panama"]

# ISO 20022 MX message types (new standard; MT103/MT202 etc. are legacy)
MESSAGE_TYPES = ["pacs.008", "pacs.009", "camt.054", "camt.053"]

def is_round_number(amount):
    """Checks if the amount is a round number, as fraud messages tend to have suspicious amounts."""
    return amount % 10000 == 0

def generate_normal_transaction(noise=0.0):
    """One legitimate-looking SWIFT (ISO 20022) transaction. With noise>0, some get fraud-like features."""
    hour = random.randint(9, 17)         # 9am to 5pm (business hours)
    day = random.randint(0, 4)            # Mon=0 .. Fri=4
    amount = random.randint(10000, 10000000)  # Normal range (1k to 10M)
    sender = random.choice(COMMON_COUNTRIES)
    receiver = random.choice(COMMON_COUNTRIES)
    account_age = random.randint(1, 3000)   # Established account
    velocity = random.randint(1, 5)           # Few tx per day
    ip_matches = random.random() < 0.95        # 95% match, True 95% of the time
    has_typos = random.random() < 0.05          # 5% typos, True 5% of the time
    msg_type = random.choice(MESSAGE_TYPES)
    if noise > 0 and random.random() < noise:
        # Confusing: look partly like fraud (e.g. off-hour or risky route or high velocity)
        if random.random() < 0.33:
            hour = random.choice([23, 0, 1, 2, 3, 4, 5])
            day = random.randint(0, 6)
        elif random.random() < 0.5:
            receiver = random.choice(RISKY_COUNTRIES)
        else:
            velocity = random.randint(10, 50)
            ip_matches = random.random() < 0.5

    return {
        "amount": amount,
        "hour_of_day": hour,
        "day_of_week": day,
        "sender_country": sender,
        "receiver_country": receiver,
        "account_age_days": account_age,
        "transaction_velocity": velocity,
        "ip_country_matches_sender": 1 if ip_matches else 0,
        "message_has_typos": 1 if has_typos else 0,
        "is_round_number": 1 if is_round_number(amount) else 0,
        "message_type": msg_type,
        "is_fraud": 0,
    }

def generate_fraudulent_transaction(noise=0.0):
    """One transaction that looks like fraud. With noise>0, some get normal-like features (confusing)."""
    # Off-hours: late night (11pm–5am) or Friday evening
    if random.random() < 0.5:               # 50% chance of late night
        hour = random.choice([23, 0, 1, 2, 3, 4, 5])   # 11pm to 5am
        day = random.randint(0, 6)
    else:
        hour = random.randint(16, 23)       # 4pm to 11pm
        day = 5                             # Friday

    amount = random.randint(50000, 9999999)
    if is_round_number(amount):             # Avoid round numbers (under threshold)
        amount -= random.randint(1, 9999)

    sender = random.choice(COMMON_COUNTRIES + RISKY_COUNTRIES)
    receiver = random.choice(RISKY_COUNTRIES)
    account_age = random.randint(1, 30)    # New or mule account
    velocity = random.randint(10, 50)       # High volume
    ip_matches = random.random() < 0.3      # 30% match
    has_typos = random.random() < 0.6       # 60% typos
    msg_type = random.choice(MESSAGE_TYPES)
    if noise > 0 and random.random() < noise:
        # Confusing: look partly normal (e.g. business hours, common countries)
        if random.random() < 0.33:
            hour = random.randint(9, 17)
            day = random.randint(0, 4)
        elif random.random() < 0.5:
            receiver = random.choice(COMMON_COUNTRIES)
            sender = random.choice(COMMON_COUNTRIES)
        else:
            account_age = random.randint(200, 1000)
            velocity = random.randint(1, 5)

    return {
        "amount": amount,
        "hour_of_day": hour,
        "day_of_week": day,
        "sender_country": sender,
        "receiver_country": receiver,
        "account_age_days": account_age,
        "transaction_velocity": velocity,
        "ip_country_matches_sender": 1 if ip_matches else 0,
        "message_has_typos": 1 if has_typos else 0,
        "is_round_number": 1 if is_round_number(amount) else 0,
        "message_type": msg_type,
        "is_fraud": 1,
    }

def generate_dataset(n_normal=900, n_fraud=100, seed=42, noise=0.0):
    """Build a table of normal + fraud transactions and shuffle. noise=0.1..0.3 adds overlapping/confusing cases."""
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    for _ in range(n_normal):
        rows.append(generate_normal_transaction(noise=noise))
    for _ in range(n_fraud):
        rows.append(generate_fraudulent_transaction(noise=noise))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Created {len(df)} rows: {(df['is_fraud'] == 0).sum()} normal, {df['is_fraud'].sum()} fraud")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic SWIFT-style data (our schema only)")
    p.add_argument("-n", "--num-rows", type=int, default=1000, help="Total rows (default 1000)")
    p.add_argument("--fraud-ratio", type=float, default=0.1, help="Fraction of fraud (default 0.1)")
    p.add_argument("-o", "--output", default="data/swift_messages.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed (use different seed for test set)")
    p.add_argument("--noise", type=float, default=0.0, help="Overlap/confusion 0-1: some normals look like fraud, some fraud like normal (e.g. 0.2)")
    args = p.parse_args()
    n_total = max(10, args.num_rows)
    n_fraud = int(round(n_total * args.fraud_ratio))
    n_normal = n_total - n_fraud
    df = generate_dataset(n_normal=n_normal, n_fraud=n_fraud, seed=args.seed, noise=args.noise)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(df.head(10))