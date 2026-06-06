"""
End-to-end test: fetch fighter data for Jon Jones and Stipe Miocic,
merge with dataset, compute features, and print results.

Run from project root:
  python -m scraper.test_scraper
"""

from datetime import date
from playwright.sync_api import sync_playwright

from scraper import ufcstats, features, data_loader


def get_fighter_data(page, name: str) -> tuple[dict, list[dict]]:
    """Return (profile_dict, sorted_fight_list) for a fighter."""
    print(f"\n--- {name} ---")

    # 1. Check local dataset first
    profile = data_loader.get_fighter_profile(name)
    history = data_loader.get_fighter_history(name)

    if profile:
        print(f"  Found in dataset: {len(history)} fights, profile URL: {profile.get('fighter_url','?')[:60]}")
    else:
        print("  Not in dataset, will scrape profile...")

    # 2. Find profile URL (from dataset or search)
    profile_url = (profile or {}).get("fighter_url")
    if not profile_url:
        profile_url = ufcstats.search_fighter(page, name)
        print(f"  Search result: {profile_url}")

    if not profile_url:
        print(f"  ERROR: could not find {name}")
        return {}, []

    # 3. If profile not in dataset, scrape it
    if not profile:
        profile = ufcstats.scrape_fighter_profile(page, profile_url)
        print(f"  Scraped profile: {profile}")

    # 4. Fetch any new fights after dataset cutoff
    new_fights = ufcstats.get_new_fights(page, profile_url, name)
    if new_fights:
        print(f"  Scraped {len(new_fights)} new fights after dataset cutoff:")
        for f in new_fights:
            print(f"    {f['date']} {f['result']} — KD:{f['knockdowns']} Sig:{f['sig_strikes']}/{f['sig_strike_atts']}")
        history = history + new_fights

    print(f"  Total fight history: {len(history)}")
    return profile, history


def main():
    fighters = [
        ("Jon Jones", "Stipe Miocic"),
        ("Islam Makhachev", "Arman Tsarukyan"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for f1_name, f2_name in fighters:
            print(f"\n{'='*60}")
            print(f"MATCHUP: {f1_name} vs {f2_name}")
            print(f"{'='*60}")

            f1_profile, f1_history = get_fighter_data(page, f1_name)
            f2_profile, f2_history = get_fighter_data(page, f2_name)

            if not f1_profile or not f2_profile:
                print("Skipping — missing profile data")
                continue

            f1_stats = features.compute_fighter_stats(f1_history)
            f2_stats = features.compute_fighter_stats(f2_history)

            print(f"\n  {f1_name}: exp={f1_stats['experience']} W={f1_stats['wins']} L={f1_stats['losses']}")
            print(f"  {f2_name}: exp={f2_stats['experience']} W={f2_stats['wins']} L={f2_stats['losses']}")

            row = features.build_feature_row(f1_stats, f1_profile, f2_stats, f2_profile)
            print(f"\n  Feature row shape: {row.shape}")
            print(f"  Columns ({len(row.columns)}): {list(row.columns[:5])} ... {list(row.columns[-3:])}")
            print(f"  diff_experience={row['diff_experience'].iloc[0]:.1f}")
            print(f"  diff_wins={row['diff_wins'].iloc[0]:.1f}")
            print(f"  diff_height_cm={row['diff_height_cm'].iloc[0]:.2f}")

        browser.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
