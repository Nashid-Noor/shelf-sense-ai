from difflib import SequenceMatcher

def test_sim(query, title):
    sim = SequenceMatcher(None, query.lower(), title.lower()).ratio()
    print(f"'{query}' vs '{title}' = {sim:.2f}")

test_sim("Java Procri", "Journal of the Asiatic Society of Bengal")
test_sim("Java Procri", "Java Programming")
test_sim("Programming in C", "Programming in C")
test_sim("Unknown Book", "Journal of the Asiatic Society of Bengal")
