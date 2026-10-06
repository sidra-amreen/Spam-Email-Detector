import os
import urllib.request
import pandas as pd

DATA_PATH = os.path.join("data", "spam.tsv")
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

FALLBACK_SPAM = [
    "WINNER!! You have been selected to receive a $1000 prize. Call now to claim",
    "Congratulations! You won a free iPhone. Click http://bit.ly/claim to collect",
    "URGENT! Your account is suspended. Verify your password immediately",
    "Free entry in our weekly draw. Text WIN to 80085 to enter",
    "Buy cheap meds online, no prescription needed. Limited time offer",
    "You have been pre-approved for a loan. Reply YES to get cash today",
    "Claim your free gift card now! Offer expires tonight, click here",
    "Earn $5000 per week working from home. No experience required",
    "Your PayPal account has been limited. Log in to restore access now",
    "Hot singles in your area want to meet you. Click to chat free",
    "Final notice: you owe back taxes. Pay now to avoid arrest",
    "Lottery winner! Send your bank details to receive your million dollars",
]
FALLBACK_HAM = [
    "Hey, are we still meeting for lunch tomorrow at 1?",
    "Can you send me the notes from today's lecture?",
    "Don't forget to pick up milk on your way home",
    "Happy birthday! Hope you have a great day with your family",
    "The meeting has been moved to 3pm in conference room B",
    "I'll call you after work, running a bit late today",
    "Thanks for dinner last night, it was really good",
    "Please find the project report attached, let me know your thoughts",
    "Did you watch the match yesterday? What a game",
    "Mom says dinner is at 8, are you coming?",
    "Can we reschedule our call to Thursday morning?",
    "Reminder: dentist appointment on Monday at 10am",
]


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        try:
            print("Downloading SMS Spam Collection...")
            urllib.request.urlretrieve(URL, DATA_PATH)
        except Exception as e:
            print(f"Download failed ({e}). Using small built-in sample instead.")
            rows = [("spam", t) for t in FALLBACK_SPAM] + [("ham", t) for t in FALLBACK_HAM]
            return pd.DataFrame(rows * 5, columns=["label", "text"])
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"]).dropna()
    return df
