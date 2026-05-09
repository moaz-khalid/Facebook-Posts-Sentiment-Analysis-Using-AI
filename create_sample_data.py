import pandas as pd
import random

positives = [
    "I love the new update! It's amazing", "Absolutely fantastic experience", 
    "Great customer service, very helpful", "So happy with this product",
    "Thanks for the wonderful evening", "I really enjoyed using this app",
    "The event was well organized", "So glad I switched to this brand",
    "I'm quite satisfied with the result", "Excellent quality and fast delivery",
    "Couldn't be happier with my purchase", "Best app I've ever used",
    "Highly recommend this to everyone", "Five stars, truly outstanding",
    "Such a user-friendly interface", "Never been this impressed",
    "A breath of fresh air in social media", "This platform keeps getting better",
    "My favorite place to connect with friends", "Outstanding performance",
    "Very reliable and intuitive", "Perfectly meets my needs",
    "I appreciate the quick response", "Way better than competitors"
]
neutrals = [
    "It's okay, nothing special", "The meeting was scheduled for Tuesday",
    "The weather is cloudy today", "I have no opinion on this matter",
    "Just an average day at work", "Not bad but could be better",
    "It works as expected", "I don't care about this update",
    "Seen better days", "Standard service, nothing to write home about",
    "The app does what it says", "Mid-tier experience overall",
    "Average quality for the price", "I'm neutral about this change",
    "Nothing groundbreaking here", "It's a tool, it functions",
    "I use it when necessary", "Neither love nor hate it",
    "Does the job, I guess", "Haven't formed an opinion yet",
    "Just another social platform", "No strong feelings either way"
]
negatives = [
    "This is the worst service ever", "I hate the new layout, so confusing",
    "Terrible experience, never coming back", "Very disappointed with the quality",
    "App keeps crashing, extremely frustrating", "Horrible, I want my money back",
    "Worst app ever, full of bugs", "Awful, just awful",
    "Absolutely useless feature", "Such a waste of time and money",
    "It's been downhill since the last update", "Customer support is non-existent",
    "I regret downloading this", "Uninstalled immediately",
    "So many ads, so little value", "Slow, buggy, and frustrating",
    "Don't trust this company", "Pathetic attempt at a redesign",
    "Ruined my whole day", "Why do I even bother with this app"
]

data = []
for text in positives:
    data.append((text, "positive"))
for text in neutrals:
    data.append((text, "neutral"))
for text in negatives:
    data.append((text, "negative"))

# Duplicate and shuffle to reach 250 samples (optional)
while len(data) < 250:
    data.extend(random.choices(data, k=250-len(data)))
random.shuffle(data)

df = pd.DataFrame(data, columns=["post_text", "sentiment"])
df.to_csv("facebook_sentiment.csv", index=False)
print(f"Created dataset with {len(df)} posts.")