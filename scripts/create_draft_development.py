"""Create transparent AI-authored development fixtures, never reviewed ground truth."""

import json
from pathlib import Path

from category_matcher.taxonomy import read_google_taxonomy

PAIRS = [
    ("Mobile Phones", "handheld cellular telephones"),
    ("Mobile Phone Cases", "protective shells for cellphones"),
    ("Laptops", "portable notebook computers"),
    ("Sofas", "living room couches"),
    ("Chairs", "seating chairs for the home"),
    ("Dog Food", "kibble for pet dogs"),
    ("Cat Toys", "playthings for cats"),
    ("Bird Cages & Stands", "enclosures for pet birds"),
    ("Shoes", "footwear"),
    ("Shirts & Tops", "upper body shirts and tops"),
    ("Handbags", "purses carried by hand"),
    ("Wedding Dresses", "bridal gowns"),
    ("Toys", "children's playthings"),
    ("Board Games", "tabletop board games"),
    ("Jigsaw Puzzles", "interlocking picture puzzles"),
    ("Bicycles", "pedal powered bicycles"),
    ("Digital Cameras", "digital photography cameras"),
    ("Camera Lenses", "interchangeable camera optics"),
    ("Baby Strollers", "pushchairs for babies"),
    ("Diapers", "baby nappies"),
    ("Baby & Toddler Clothing", "clothes for infants and toddlers"),
    ("Coffee", "coffee beverages"),
    ("Apples", "fresh apples"),
    ("Screwdrivers", "tools for turning screws"),
    ("Hammers", "hand tools for driving nails"),
    ("Lawn Mowers", "machines for cutting lawn grass"),
    ("Garden Hoses", "flexible hoses for watering gardens"),
    ("Beds & Bed Frames", "bed frames for sleeping"),
    ("Toothbrushes", "brushes for cleaning teeth"),
    ("Shampoo & Conditioner", "hair shampoo and conditioner"),
    ("Sunscreen", "sun protection skin lotion"),
    ("Backpacks", "bags worn on the back"),
    ("Office Chairs", "desk seating for office workers"),
    ("Staplers", "paper fastening staplers"),
    ("Antivirus & Security Software", "software that protects against computer viruses"),
    ("Guitars", "six string guitars"),
    ("Paint Brushes", "brushes for applying paint"),
    ("Industrial Storage", "storage equipment for industrial facilities"),
    ("Sex Toys", "adult erotic toys"),
    ("Books", "printed reading books"),
    ("DVDs & Videos", "movies on DVD"),
    ("Cars, Trucks & Vans", "passenger cars trucks and vans"),
    ("Bicycle Helmets", "protective helmets for bicycle riders"),
    ("Computer Monitors", "desktop computer displays"),
    ("Headphones & Headsets", "personal audio headphones and headsets"),
    ("Fish Food", "feed for aquarium fish"),
    ("Musical Instruments", "instruments used to make music"),
    ("Office Supplies", "supplies for office work"),
    ("Religious & Ceremonial", "religious and ceremonial goods"),
    ("Business & Industrial", "commercial and industrial goods"),
    ("Cameras & Optics", "cameras and optical equipment"),
    ("Food, Beverages & Tobacco", "food drinks and tobacco products"),
    ("Health & Beauty", "health and beauty products"),
    ("Home & Garden", "household and gardening goods"),
    ("Vehicles & Parts", "vehicles and vehicle parts"),
    ("Luggage & Bags", "travel luggage and bags"),
    ("Mature", "adult oriented products"),
    ("Hardware", "building hardware and tools"),
    ("Media", "media products"),
    ("Baby & Toddler", "goods for babies and toddlers"),
]

HIERARCHY = [
    ("Animals & Pet Supplies", "all animal and pet products"),
    ("Apparel & Accessories", "clothing and fashion accessories in general"),
    ("Arts & Entertainment", "arts crafts and entertainment products"),
    ("Baby & Toddler", "baby and toddler products in general"),
    ("Business & Industrial", "business and industrial equipment in general"),
    ("Cameras & Optics", "photography and optics equipment in general"),
    ("Electronics", "electronic devices in general"),
    ("Food, Beverages & Tobacco", "food beverages and tobacco in general"),
    ("Furniture", "furniture in general"),
    ("Hardware", "hardware products in general"),
    ("Health & Beauty", "health and beauty goods in general"),
    ("Home & Garden", "home and garden products in general"),
    ("Luggage & Bags", "luggage and bags in general"),
    ("Mature", "mature adult products in general"),
    ("Media", "media content products in general"),
    ("Office Supplies", "office supplies in general"),
    ("Religious & Ceremonial", "religious and ceremonial products in general"),
    ("Software", "software products in general"),
    ("Sporting Goods", "sports equipment in general"),
    ("Toys & Games", "toys and games in general"),
    ("Vehicles & Parts", "vehicles and their parts in general"),
    ("Pet Supplies", "supplies for pets in general"),
    ("Dog Supplies", "supplies for dogs in general"),
    ("Cat Supplies", "supplies for cats in general"),
    ("Mobile Phone Accessories", "phone accessories"),
    ("Computers", "computers in general"),
    ("Clothing", "clothing in general"),
    ("Tools", "tools in general"),
]


def main() -> None:
    categories, audit = read_google_taxonomy(Path("google_category_list.txt"))
    labels = {category.label: category for category in categories}
    records = []

    def add(query, slice_name, label=None, *, category_id=None, status="matched", language="en"):
        records.append(
            {
                "example_id": f"draft-{len(records) + 1:03d}",
                "taxonomy_key": "google",
                "taxonomy_version": audit["source_sha256"][:16],
                "query": query,
                "category_id": category_id,
                "context": "",
                "parent_category_id": None,
                "language": language,
                "slice": slice_name,
                "expected_status": status,
                "acceptable_category_ids": [labels[label].category_id] if label else [],
                "input_family": label or query,
                "source": "ai_authored_development_fixture",
                "review_status": "pending_review",
                "reviewer": None,
                "reviewed_at": None,
            }
        )

    for category in [c for c in categories if c.depth == 1]:
        add(category.label, "exact", category.label)
    for index, (label, _) in enumerate(PAIRS[:19]):
        if index % 2:
            add("", "exact", label, category_id=labels[label].category_id)
        else:
            add(labels[label].full_path, "exact", label)
    for label, paraphrase in PAIRS:
        add(paraphrase, "paraphrase", label)
    for label, _ in PAIRS[:32]:
        # A deterministic dropped internal character; always marked synthetic.
        index = next(i for i, char in enumerate(label) if i >= 2 and char.isalpha())
        add(label[:index] + label[index + 1 :], "typo", label)
    for label, paraphrase in HIERARCHY:
        add(paraphrase, "hierarchy", label)
    for term in [
        "apple",
        "charger",
        "adapter",
        "case",
        "cover",
        "stand",
        "rack",
        "tablet",
        "mouse",
        "bass",
        "drum",
        "brush",
        "cable",
        "filter",
        "battery",
        "pump",
        "bag",
        "cleaner",
        "holder",
        "pad",
    ]:
        add(term, "ambiguous", status="needs_clarification")
    for query in [
        "legal consultation services",
        "tax accounting service",
        "dental surgery appointment",
        "home cleaning service",
        "online math tutoring",
        "car insurance policy",
        "hotel room reservation",
        "restaurant table booking",
        "cloud server rental",
        "mobile app development service",
        "dog grooming appointment",
        "home moving service",
        "zxqvplm",
        "qwertyuiopxyz",
        "zzzzvvvqqq",
        "asdfghjklzzz",
    ]:
        add(query, "no_match", status="no_match")
    for query, language in [
        ("মোবাইল ফোন", "bn"),
        ("teléfonos móviles", "es"),
        ("chaussures", "fr"),
        ("Mobiltelefone", "de"),
    ]:
        add(query, "unsupported_language", status="unsupported_language", language=language)
    assert len(records) == 200
    output = Path("data/google/evaluation/development.jsonl")
    if output.exists():
        raise SystemExit("Refusing to overwrite existing labels or reviewer edits")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(f"Wrote {len(records)} UNREVIEWED development fixtures to {output}")


if __name__ == "__main__":
    main()
