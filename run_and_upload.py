#!/usr/bin/env python3
"""
=============================================================================
RUN & UPLOAD SCRIPT
Runs the MLP MVP bot and uploads the submission to the competition dashboard
=============================================================================
"""

import subprocess
import sys
import os
import time
import re
from pathlib import Path

# Configuration
BOT_SCRIPT = "bot_knn_cash_cow/knn_cash_cow.py"
SUBMISSION_FILE = "submissions/my_team_name_knn_submission.joblib"
DASHBOARD_URL = "https://quant-trading.lovable.app/"
TEAM_NAME = "3D1I"

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()


def install_selenium():
    """Install selenium if not present."""
    try:
        import selenium
        print("✓ Selenium is already installed")
    except ImportError:
        print("📦 Installing selenium...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
        print("✓ Selenium installed successfully")


def run_bot():
    """Run the MLP MVP bot to generate the submission file."""
    print("\n" + "=" * 60)
    print("🤖 STEP 1: Running MLP MVP Bot")
    print("=" * 60)
    
    bot_path = PROJECT_ROOT / BOT_SCRIPT
    
    if not bot_path.exists():
        print(f"❌ Error: Bot script not found at {bot_path}")
        return False
    
    # Run the bot script
    result = subprocess.run(
        [sys.executable, str(bot_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ Error: Bot script failed with return code {result.returncode}")
        return False
    
    # Check if submission file was created
    submission_path = PROJECT_ROOT / SUBMISSION_FILE
    if not submission_path.exists():
        print(f"❌ Error: Submission file not found at {submission_path}")
        return False
    
    print(f"\n✓ Submission file created: {submission_path}")
    print(f"  File size: {submission_path.stat().st_size / 1024:.2f} KB")
    return True


def upload_submission():
    """Open browser and upload the submission file."""
    print("\n" + "=" * 60)
    print("📤 STEP 2: Uploading to Competition Dashboard")
    print("=" * 60)
    
    # Install selenium if needed
    install_selenium()
    
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        use_webdriver_manager = True
    except ImportError:
        use_webdriver_manager = False
    
    submission_path = str((PROJECT_ROOT / SUBMISSION_FILE).absolute())
    
    print(f"📂 Submission file: {submission_path}")
    print(f"🌐 Opening {DASHBOARD_URL}")
    
    # Setup Chrome options
    chrome_options = Options()
    # Keep browser open after script ends
    chrome_options.add_experimental_option("detach", True)
    
    try:
        # Try to use webdriver-manager first
        if use_webdriver_manager:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            driver = webdriver.Chrome(options=chrome_options)
        
        driver.get(DASHBOARD_URL)
        print("✓ Browser opened successfully")
        
        # Wait for page to load
        time.sleep(3)
        
        # Try to find file input element and upload
        try:
            # Common selectors for file upload inputs
            file_input_selectors = [
                "input[type='file']",
                "input[accept='.joblib']",
                "input[accept='*']",
                "#file-upload",
                ".file-input",
                "input[name='file']",
                "input[name='submission']"
            ]
            
            file_input = None
            for selector in file_input_selectors:
                try:
                    file_input = WebDriverWait(driver, 2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if file_input:
                        print(f"✓ Found file input with selector: {selector}")
                        break
                except:
                    continue
            
            if file_input:
                # Upload the file
                file_input.send_keys(submission_path)
                print("✓ File selected for upload!")
                
                # Try to fill in team name
                try:
                    team_name_selectors = [
                        "input[placeholder*='team']",
                        "input[placeholder*='Team']",
                        "input[name='team']",
                        "input[name='teamName']",
                        "input[id*='team']",
                        "input[type='text']"
                    ]
                    
                    team_input = None
                    for selector in team_name_selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for elem in elements:
                                if elem.is_displayed() and elem.is_enabled():
                                    team_input = elem
                                    break
                            if team_input:
                                break
                        except:
                            continue
                    
                    if team_input:
                        team_input.clear()
                        team_input.send_keys(TEAM_NAME)
                        print(f"✓ Team name set to: {TEAM_NAME}")
                except Exception as e:
                    print(f"⚠️  Could not auto-fill team name: {e}")
                
                # Try to click submit button
                time.sleep(1)
                try:
                    submit_selectors = [
                        "button[type='submit']",
                        "button:contains('Submit')",
                        "button:contains('Upload')",
                        "input[type='submit']",
                        "button.submit",
                        "button"
                    ]
                    
                    submit_btn = None
                    for selector in submit_selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for elem in elements:
                                text = elem.text.lower()
                                if elem.is_displayed() and elem.is_enabled() and ('submit' in text or 'upload' in text):
                                    submit_btn = elem
                                    break
                            if submit_btn:
                                break
                        except:
                            continue
                    
                    if submit_btn:
                        print(f"✓ Found submit button: '{submit_btn.text}'")
                        submit_btn.click()
                        print("✓ Submit button clicked!")
                        print("⏳ Waiting for evaluation result...")
                        
                        # Wait for evaluation result
                        alpha_value = None
                        for attempt in range(30):  # Wait up to 30 seconds
                            time.sleep(1)
                            try:
                                # Look for the result text containing "alpha"
                                page_text = driver.find_element(By.TAG_NAME, "body").text
                                
                                # Search for alpha pattern like "alpha: X.XX%" or "Alpha: X.XX%"
                                alpha_match = re.search(r'alpha[:\s]+(-?\d+\.?\d*)%', page_text, re.IGNORECASE)
                                if alpha_match:
                                    alpha_value = alpha_match.group(1)
                                    print(f"\n{'='*60}")
                                    print(f"🎯 EVALUATION RESULT: Alpha = {alpha_value}%")
                                    print(f"{'='*60}")
                                    break
                                
                                # Also look for rank
                                rank_match = re.search(r'Rank[:\s#]+(\d+)', page_text, re.IGNORECASE)
                                if rank_match and alpha_match:
                                    print(f"📊 Rank: #{rank_match.group(1)}")
                                    
                            except Exception:
                                pass
                            
                            if attempt % 5 == 4:
                                print(f"   Still waiting... ({attempt+1}s)")
                        
                        if not alpha_value:
                            print("⚠️  Could not capture alpha value automatically")
                            print("   Check the browser for the result")
                        
                        # Close browser after getting result
                        time.sleep(2)
                        driver.quit()
                        print("✓ Browser closed")
                        return alpha_value
                        
                    else:
                        print("⚠️  Could not find submit button - please click it manually")
                except Exception as e:
                    print(f"⚠️  Could not auto-click submit: {e}")
                
            else:
                print("\n⚠️  Could not find file upload input automatically.")
                print("   The browser is open - please upload manually:")
                print(f"   File to upload: {submission_path}")
                print(f"   Team name: {TEAM_NAME}")
                
        except Exception as e:
            print(f"\n⚠️  Could not auto-upload: {e}")
            print("   The browser is open - please upload manually:")
            print(f"   File to upload: {submission_path}")
        
        # If we get here, something went wrong - keep browser open for manual interaction
        print("\n" + "=" * 60)
        print("🎯 Browser is open - complete manually if needed")
        print(f"   Team name: {TEAM_NAME}")
        print(f"   File: {submission_path}")
        print("=" * 60)
        
        print("\nPress Enter to close the browser...")
        input()
        driver.quit()
        
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("\n💡 Manual upload instructions:")
        print(f"   1. Open: {DASHBOARD_URL}")
        print(f"   2. Upload: {submission_path}")
        return False
    
    return True


def main():
    """Main function to run bot and upload submission."""
    print("\n" + "=" * 60)
    print("🚀 VIBETRADING - RUN & UPLOAD SCRIPT")
    print("=" * 60)
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"👥 Team name: {TEAM_NAME}")
    
    # Step 1: Run the bot
    if not run_bot():
        print("\n❌ Failed to run bot. Exiting.")
        sys.exit(1)
    
    # Step 2: Upload the submission and get alpha
    alpha = upload_submission()
    
    if alpha:
        print(f"\n✅ Done! Final Alpha: {alpha}%")
    else:
        print("\n✅ Done!")


if __name__ == "__main__":
    main()

