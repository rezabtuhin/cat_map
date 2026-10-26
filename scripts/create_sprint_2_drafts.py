"""Prepare disjoint review proposals, not trusted calibration/test ground truth."""

import json
import random
from pathlib import Path

from category_matcher.datasets import check_splits, read_jsonl
from category_matcher.normalization import exact_key
from category_matcher.taxonomy import read_google_taxonomy

PAIRS = [
    ("Refrigerators", "appliances for keeping food cold"),
    ("Microwave Ovens", "microwave food heating appliances"),
    ("Ovens", "ovens for baking food"),
    ("Dishwashers", "automatic dish washing machines"),
    ("Food Mixers & Blenders", "kitchen mixing and blending appliances"),
    ("Toasters", "appliances for toasting bread"),
    ("Freezers", "appliances that keep food frozen"),
    ("Vacuums", "vacuum cleaning appliances"),
    ("Washing Machines", "machines for washing laundry"),
    ("Dryers", "appliances for drying clothes"),
    ("Irons & Ironing Systems", "equipment for ironing clothes"),
    ("Bowls", "bowls for serving food"),
    ("Plates", "plates for serving meals"),
    ("Mugs", "handled mugs for hot drinks"),
    ("Skillets & Frying Pans", "pans for frying food"),
    ("Saucepans", "handled pots for cooking sauces"),
    ("Stock Pots", "large cooking pots for stock"),
    ("Cutting Boards", "boards for chopping food"),
    ("Toilet Paper", "bathroom tissue rolls"),
    ("Paper Towels", "absorbent kitchen paper rolls"),
    ("Trash Cans & Wastebaskets", "bins for household rubbish"),
    ("Brooms", "brushes for sweeping floors"),
    ("Mops", "tools for wet mopping floors"),
    ("Laundry Detergent", "detergent for washing clothes"),
    ("Dish Detergent & Soap", "soap for washing dishes"),
    ("Bar Soap", "solid bars of body soap"),
    ("Liquid Hand Soap", "liquid soap for washing hands"),
    ("Deodorant & Anti-Perspirant", "underarm odor and perspiration products"),
    ("Lipstick", "colored lip makeup sticks"),
    ("Mascara", "makeup for eyelashes"),
    ("Nail Polishes", "colored coatings for fingernails"),
    ("Hair Dryers", "electric blow dryers for hair"),
    ("Combs & Brushes", "combs and brushes for hair grooming"),
    ("Razors & Razor Blades", "razors and blades for shaving"),
    ("Shaving Cream", "cream for shaving facial hair"),
    ("Perfume & Cologne", "fragrances worn on the body"),
    ("Medical Tape & Bandages", "medical bandages and adhesive tape"),
    ("First Aid Kits", "kits of emergency first aid supplies"),
    ("Medical Thermometers", "thermometers for measuring body temperature"),
    ("Body Weight Scales", "scales for weighing a person"),
    ("Blood Pressure Monitors", "devices for checking blood pressure"),
    ("Eyeglasses", "prescription spectacles"),
    ("Contact Lenses", "corrective lenses worn on the eye"),
    ("Hearing Aids", "devices that help people hear"),
    ("Canes & Walking Sticks", "sticks that assist walking"),
    ("Wheelchairs", "chairs with wheels for mobility"),
    ("Watches", "wrist worn timepieces"),
    ("Rings", "jewelry worn on fingers"),
    ("Necklaces", "jewelry worn around the neck"),
    ("Earrings", "jewelry worn on ears"),
    ("Bracelets", "jewelry worn around the wrist"),
    ("Wallets & Money Clips", "wallets and clips for carrying money"),
    ("Belts", "belts worn around the waist"),
    ("Gloves & Mittens", "hand covering apparel"),
    ("Hats", "hats worn on the head"),
    ("Sunglasses", "sun protective eyewear"),
    ("Parasols & Rain Umbrellas", "handheld rain umbrellas and parasols"),
    ("Suitcases", "travel suitcases"),
    ("Briefcases", "business document carrying cases"),
    ("Sleeping Bags", "sleeping bags for camping"),
    ("Tents", "portable camping shelters"),
    ("Flashlights", "handheld electric torches"),
    ("Binoculars", "two eyepiece distance viewing optics"),
    ("Telescopes", "optical instruments for viewing distant objects"),
    ("Tripods & Monopods", "three legged and single leg camera supports"),
    ("Microphones", "devices for recording voices"),
    ("Speakers", "audio loudspeakers"),
    ("Audio Amplifiers", "amplifiers for audio signals"),
    ("Televisions", "television sets"),
    ("Multimedia Projectors", "projectors for computer presentations and video"),
    ("Tablet Computers", "touchscreen tablet computers"),
    ("Desktop Computers", "stationary personal computers"),
    ("Mice & Trackballs", "computer pointing devices"),
    ("Keyboards", "keyboards for typing on computers"),
    ("Memory", "computer memory modules"),
    ("Hard Drives", "hard disk storage drives"),
    ("USB Flash Drives", "USB memory sticks"),
    ("Memory Card Readers", "devices for reading memory cards"),
    ("Flash Memory Cards", "removable flash memory cards"),
    ("Scanners", "devices for scanning paper documents"),
    ("Printers, Copiers & Fax Machines", "printing copying and fax devices"),
    ("Waffle Irons", "appliances for making waffles"),
]

AMBIGUOUS = [
    "accessory",
    "replacement",
    "parts",
    "support",
    "kit",
    "set",
    "device",
    "equipment",
    "supplies",
    "tools",
    "mount",
    "base",
    "insert",
    "refill",
    "attachment",
    "frame",
    "sleeve",
    "liner",
    "cap",
    "tip",
    "sheet",
    "roll",
    "panel",
    "tube",
    "port",
    "head",
    "lead",
    "ring",
    "bar",
    "block",
    "box",
    "pack",
    "clip",
    "connector",
    "switch",
    "control",
    "sensor",
    "module",
    "receiver",
    "transmitter",
    "converter",
    "extension",
    "cushion",
    "mat",
    "net",
    "strap",
    "seal",
    "valve",
    "spout",
    "tray",
    "plate",
    "screen",
    "light",
    "fan",
    "heater",
    "cooler",
    "scale",
    "meter",
    "drive",
    "disc",
]

SERVICES = [
    "legal advice",
    "tax filing",
    "accounting consultation",
    "dental appointment",
    "medical appointment",
    "house cleaning",
    "car insurance",
    "health insurance",
    "life insurance",
    "hotel booking",
    "restaurant reservation",
    "flight booking",
    "train ticket reservation",
    "bus ticket reservation",
    "cloud hosting rental",
    "software development",
    "website design",
    "logo design",
    "translation service",
    "proofreading service",
    "resume writing",
    "business consulting",
    "career counseling",
    "financial planning",
    "mortgage advice",
    "real estate valuation",
    "property management",
    "plumbing repair service",
    "electrical repair service",
    "roof repair service",
    "window cleaning service",
    "landscape maintenance service",
    "lawn mowing service",
    "pest control service",
    "security patrol service",
    "private tutoring",
    "language tutoring",
    "music lesson booking",
    "personal training service",
    "fitness class booking",
    "massage appointment",
    "haircut appointment",
    "beauty treatment booking",
    "pet sitting",
    "dog walking service",
    "cat boarding reservation",
    "furniture moving service",
    "parcel delivery service",
    "vehicle towing service",
    "automobile rental booking",
    "wedding planning service",
    "event catering service",
]


def main():
    base = Path("data/google/evaluation")
    paths = {name: base / f"{name}.draft.jsonl" for name in ("calibration", "test")}
    if any(path.exists() for path in paths.values()):
        raise SystemExit("Refusing to overwrite existing labels or reviewer edits")
    categories, audit = read_google_taxonomy(Path("google_category_list.txt"))
    by_label = {c.label: c for c in categories}
    used = {
        cid
        for row in read_jsonl(base / "development.jsonl") + read_jsonl(base / "user_examples.jsonl")
        for cid in row["acceptable_category_ids"]
    }
    pairs = [
        (by_label[label], query)
        for label, query in PAIRS
        if label in by_label and by_label[label].category_id not in used
    ][:80]
    if len(pairs) != 80:
        raise ValueError("Need 80 new valid category paraphrase proposals")
    used.update(category.category_id for category, _ in pairs)
    pool = [c for c in categories if c.category_id not in used]
    random.Random(20261026).shuffle(pool)
    output = {"calibration": [], "test": []}

    def add(split, query, slice_name, category=None, status="matched", language="en"):
        rows = output[split]
        rows.append(
            {
                "example_id": f"s2-{split}-{len(rows) + 1:03d}",
                "taxonomy_key": "google",
                "taxonomy_version": audit["source_sha256"][:16],
                "query": query,
                "context": "",
                "parent_category_id": None,
                "language": language,
                "slice": slice_name,
                "expected_status": status,
                "acceptable_category_ids": [category.category_id] if category else [],
                "input_family": f"category:{category.category_id}"
                if category
                else f"{split}:{query}",
                "source": "ai_authored_review_proposal",
                "dataset_role": f"{split}_proposal",
                "review_status": "pending_review",
                "reviewer": None,
                "reviewed_at": None,
            }
        )

    for i, (category, query) in enumerate(pairs):
        add("calibration" if i % 2 == 0 else "test", query, "paraphrase", category)
    for split in output:
        for _ in range(30):
            category = pool.pop()
            used.add(category.category_id)
            add(split, category.label, "exact", category)
        for _ in range(34):
            while True:
                category = pool.pop()
                if len(category.label) > 4 and any(char.isalpha() for char in category.label[2:]):
                    index = next(
                        i for i, char in enumerate(category.label) if i >= 2 and char.isalpha()
                    )
                    query = category.label[:index] + category.label[index + 1 :]
                    if exact_key(query) not in {exact_key(c.label) for c in categories}:
                        break
            used.add(category.category_id)
            add(split, query, "typo", category)
        for _ in range(36):
            category = next(c for c in pool if not c.is_leaf)
            pool.remove(category)
            used.add(category.category_id)
            add(split, "general " + category.label.lower() + " category", "hierarchy", category)
    for i, query in enumerate(AMBIGUOUS):
        # Avoid asserting that an existing canonical category label is ambiguous.
        if exact_key(query) in {exact_key(c.label) for c in categories}:
            query = "general " + query
        add(
            "calibration" if i % 2 == 0 else "test",
            query,
            "ambiguous",
            status="needs_clarification",
        )
    for i, service in enumerate(SERVICES):
        add(
            "calibration" if i % 2 == 0 else "test",
            service + " appointment request",
            "no_match",
            status="no_match",
        )
    for i, (query, language) in enumerate(
        [
            ("চেয়ার", "bn"),
            ("refrigerador", "es"),
            ("ordinateur portable", "fr"),
            ("Kopfhörer", "de"),
            ("ল্যাপটপ", "bn"),
            ("sillón", "es"),
            ("machine à laver", "fr"),
            ("Tastatur", "de"),
        ]
    ):
        add(
            "calibration" if i % 2 == 0 else "test",
            query,
            "unsupported_language",
            status="unsupported_language",
            language=language,
        )
    for split, rows in output.items():
        if len(rows) != 200:
            raise ValueError(f"Expected 200 {split} proposals, got {len(rows)}")
        paths[split].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report = check_splits({"development": base / "development.jsonl", **paths})
    target = Path("reports/sprint_2/data_splits.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(report["conflicts"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
