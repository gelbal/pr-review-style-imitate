import os
import json
import subprocess
from datetime import datetime
from typing import List

MAX_DIFF_LINES = 2000  # Maximum number of changed lines we'll accept

def run_shell_command(command: List[str], cwd: str = None) -> str:
    """Run a shell command and return output"""
    try:
        print(f"Running git command in {cwd}: {' '.join(command)}")  # Debug line
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            shell=False
        )

        # Check if command failed
        if result.returncode != 0:
            print(f"Error running command {' '.join(command)}")
            print(f"Error output: {result.stderr}")
            return None

        # Command succeeded, return stdout (might be empty for some git commands)
        return result.stdout.strip()

    except Exception as e:
        print(f"Exception running command {' '.join(command)}: {e}")
        return None

def get_project_root() -> str:
    """Get project root path consistently"""
    current_dir = os.getcwd()
    return os.path.abspath(os.path.join(current_dir, '..', '..')) if current_dir.endswith('notebooks') else current_dir

def get_data_dir(repo_name: str, data_type: str = "input") -> str:
    """Get data directory path"""
    project_root = get_project_root()
    return os.path.join(project_root, 'data', data_type, repo_name)

def get_work_dir(repo_name: str) -> str:
    """Get repository-specific work directory"""
    project_root = get_project_root()
    return os.path.join(project_root, 'data', 'temp_git_workspace', repo_name)

def get_diff_size(diff_content: str) -> int:
    """Count the number of changed lines in diff content"""
    if not diff_content:
        return 0
    return len([line for line in diff_content.split('\n') if line.startswith('+') or line.startswith('-')])

def update_git_repo(repo_name: str, repo_url: str) -> str:
    """Update existing git repository with latest changes"""
    work_dir = get_work_dir(repo_name)

    print(f"Updating repository in: {work_dir}")

    # If repo doesn't exist, clone it first
    if not os.path.exists(os.path.join(work_dir, '.git')):
        print("Repository not found. Performing initial clone...")
        os.makedirs(work_dir, exist_ok=True)
        clone_result = run_shell_command(['git', 'clone', repo_url, '.'], work_dir)
        if not clone_result and not os.path.exists(os.path.join(work_dir, '.git')):
            raise Exception("Failed to clone repository")

    # Update existing repo
    print("Fetching latest changes...")

    # First checkout master to ensure we're on the right branch
    checkout_result = run_shell_command(['git', 'checkout', 'master'], work_dir)
    if not checkout_result:
        raise Exception("Failed to checkout master branch")

    # Then fetch latest changes
    fetch_result = run_shell_command(['git', 'fetch', 'origin', 'master'], work_dir)
    if fetch_result is None:  # None indicates command failed
        raise Exception("Failed to fetch latest changes")

    # Reset to match origin/master
    reset_result = run_shell_command(['git', 'reset', '--hard', 'origin/master'], work_dir)
    if not reset_result:
        raise Exception("Failed to reset to origin/master")

    return work_dir

def load_and_merge_pr_data(repo_name: str) -> List[dict]:
    """Load and merge PR data from all input files"""
    input_dir = get_data_dir(repo_name, "input")

    # Load all input files
    with open(os.path.join(input_dir, "pull_requests.json"), 'r') as f:
        pull_requests = json.load(f)
    with open(os.path.join(input_dir, "pull_request_reviews.json"), 'r') as f:
        reviews = json.load(f)
    with open(os.path.join(input_dir, "pull_request_review_comments.json"), 'r') as f:
        review_comments = json.load(f)

    # Create lookup dictionaries for reviews and comments by PR ID
    reviews_by_pr = {}
    for review in reviews:
        pr_id = review['pr_id']
        if pr_id not in reviews_by_pr:
            reviews_by_pr[pr_id] = []
        reviews_by_pr[pr_id].append(review)

    comments_by_pr = {}
    for comment in review_comments:
        pr_id = comment['pr_id']
        if pr_id not in comments_by_pr:
            comments_by_pr[pr_id] = []
        comments_by_pr[pr_id].append(comment)

    # Merge data into pull requests
    for pr in pull_requests:
        pr_id = pr['pr_id']

        # Add reviews
        pr['reviews'] = reviews_by_pr.get(pr_id, [])
        pr['review_comments'] = comments_by_pr.get(pr_id, [])

        # Extract reviewers and review requesters
        reviewers = set()
        for review in pr['reviews']:
            reviewers.add(review['user_login'])
        for comment in pr['review_comments']:
            reviewers.add(comment['user_login'])

        # Get requested reviewers (assuming it's in the original PR data)
        requested_reviewers = set(pr.get('requested_reviewers', []))

        # Remove actual reviewers from requested list
        pr['reviewers'] = list(reviewers)
        pr['review_requested'] = list(requested_reviewers - reviewers)

        # Simplify labels to just names
        pr['label_names'] = [label['name'] for label in pr.get('labels', [])]

        # Reset confidentiality if PR was updated
        if 'confidentiality' in pr:
            pr['confidentiality']['is_confidential'] = None
            pr['confidentiality']['classified_at'] = None

    return pull_requests

def needs_metadata_update(existing_pr: dict, new_pr: dict) -> bool:
    """Determine if PR metadata needs to be updated"""
    if not existing_pr:
        return True

    # Check if reviews or comments have changed
    existing_reviews = len(existing_pr.get('reviews', []))
    existing_comments = len(existing_pr.get('review_comments', []))
    new_reviews = len(new_pr.get('reviews', []))
    new_comments = len(new_pr.get('review_comments', []))

    if existing_reviews != new_reviews or existing_comments != new_comments:
        return True

    # Check timestamps
    existing_updated = datetime.fromisoformat(existing_pr['updated_at'].replace('Z', ''))
    new_updated = datetime.fromisoformat(new_pr['updated_at'].replace('Z', ''))

    return new_updated > existing_updated

def needs_diff_update(existing_pr: dict, force_update: bool = False) -> bool:
    """Determine if PR diff needs to be updated"""
    if force_update:
        return True

    # Check if diff_content is missing or null
    if 'diff_content' not in existing_pr or existing_pr['diff_content'] is None:
        return True

    # Check if PR is merged and has merge commit but no diff
    if existing_pr.get('merged_at') and existing_pr.get('merge_commit_sha'):
        return not existing_pr.get('diff_content')

    return False

def fetch_pr_diff(pr_url: str, repo_name: str, merged_at: str = None, merge_commit_sha: str = None, work_dir: str = None) -> str:
    """Fetch PR diff using merge commit or latest commit for unmerged PRs"""
    pr_number = pr_url.split('/')[-1]

    try:
        if not merge_commit_sha:
            print(f"  No commit SHA available for PR #{pr_number}")
            return None

        print(f"  Fetching commit: {merge_commit_sha}")

        # First try to fetch the specific commit
        fetch_result = run_shell_command(['git', 'fetch', 'origin', merge_commit_sha], work_dir)
        if fetch_result is None:
            print(f"  Failed to fetch commit {merge_commit_sha}")
            return None

        # Get the commit's parents
        parents = run_shell_command(['git', 'rev-list', '--parents', '-n', '1', merge_commit_sha], work_dir)
        if not parents:
            print(f"  Failed to get parents for commit {merge_commit_sha}")
            return None

        parents = parents.strip().split()
        print(f"  Found {len(parents)} parent commits for {merge_commit_sha}")
        print(f"  Parent commits: {parents}")

        if len(parents) >= 2:
            current_commit = parents[0]  # The current/merge commit
            base_commit = parents[1]     # The target branch (usually master/main)

            # Try different diff combinations to find the most reasonable one
            diff_options = []

            # For both merged and unmerged PRs, try base to current commit diff
            diff_content = run_shell_command(
                ['git', 'diff', base_commit, current_commit],
                work_dir
            )
            if diff_content:
                diff_size = get_diff_size(diff_content)
                print(f"  Base to current diff size: {diff_size} lines")
                if diff_size <= MAX_DIFF_LINES:
                    diff_options.append((diff_content, diff_size, "standard"))

            # For merge commits with more than 2 parents, try PR branch to base
            if len(parents) > 2 and merged_at:
                pr_commit = parents[2]  # The PR branch
                diff_content = run_shell_command(
                    ['git', 'diff', base_commit, pr_commit],
                    work_dir
                )
                if diff_content:
                    diff_size = get_diff_size(diff_content)
                    print(f"  PR branch diff size: {diff_size} lines")
                    if diff_size <= MAX_DIFF_LINES:
                        diff_options.append((diff_content, diff_size, "pr_branch"))

            if diff_options:
                # Choose the smallest reasonable diff
                diff_options.sort(key=lambda x: x[1])  # Sort by diff size
                chosen_diff, size, diff_type = diff_options[0]
                print(f"  Choosing {diff_type} diff with {size} lines")
                return chosen_diff
            else:
                print(f"  All diff options exceeded maximum size of {MAX_DIFF_LINES} lines")
                return None
        else:
            print(f"  Error: Not enough parent commits found for commit {merge_commit_sha}")

        return None

    except Exception as e:
        print(f"Error processing PR {pr_number}: {e}")
        print(f"Exception details:", str(e))
        return None

def process_prs(repo_name: str, batch_size: int = 100, force_update: bool = False):
    """Process all PRs and fetch their diffs, saving in batches"""
    # Read and merge input PRs
    all_prs = load_and_merge_pr_data(repo_name)

    # Create output directory and load existing data
    output_dir = get_data_dir(repo_name, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{repo_name}_enriched_prs.json")

    pr_lookup = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_prs = json.load(f)
            pr_lookup = {pr['html_url']: pr for pr in existing_prs}

    # First pass: Update metadata only when needed
    metadata_updates = 0
    for pr in all_prs:
        if pr['html_url'] in pr_lookup:
            existing_pr = pr_lookup[pr['html_url']]
            if not needs_metadata_update(existing_pr, pr):
                continue

            # Preserve existing diff data
            pr['diff_content'] = existing_pr.get('diff_content')
            pr['diff_size'] = existing_pr.get('diff_size', 0)
            metadata_updates += 1
        else:
            # Initialize empty diff data for new PRs
            pr['diff_content'] = None
            pr['diff_size'] = 0
            metadata_updates += 1

        # Update PR in lookup
        pr_lookup[pr['html_url']] = pr

    print(f"Updated metadata for {metadata_updates} PRs")

    # Second pass: Process diffs only for PRs that need them
    valid_prs = []
    skipped_prs = []
    for pr in all_prs:
        if pr['html_url'] in pr_lookup:
            existing_pr = pr_lookup[pr['html_url']]
            if needs_diff_update(existing_pr, force_update):
                valid_prs.append(pr)
            else:
                skipped_prs.append(pr)
        else:
            skipped_prs.append(pr)

    print(f"Found {len(valid_prs)} PRs needing diff updates")
    print(f"Found {len(skipped_prs)} PRs to skip")
    print(f"Total PRs in system: {len(pr_lookup)}")

    # Initialize/update git repo only if we have diffs to process
    work_dir = None
    if valid_prs:
        repo_url = '/'.join(valid_prs[0]['html_url'].split('/')[:5])
        work_dir = update_git_repo(repo_name, repo_url)

    # Process diffs for valid PRs in batches
    current_batch = []
    total_prs = len(valid_prs)
    processed = 0
    successful = 0
    failed = 0

    for i, pr in enumerate(valid_prs, 1):
        try:
            print(f"Processing PR #{pr['html_url'].split('/')[-1]} ({i}/{total_prs})")
            print(f"  Merged at: {pr['merged_at']}, Merge commit: {pr['merge_commit_sha']}")

            diff_content = fetch_pr_diff(
                pr['html_url'],
                repo_name,
                pr['merged_at'],
                pr['merge_commit_sha'],
                work_dir
            )

            if diff_content:
                diff_size = get_diff_size(diff_content)
                if diff_size > MAX_DIFF_LINES:
                    print(f"  Warning: Diff size ({diff_size} lines) exceeds maximum ({MAX_DIFF_LINES})")
                    failed += 1
                    continue

                # Update PR with diff content
                enriched_pr = pr_lookup[pr['html_url']]
                enriched_pr['diff_content'] = diff_content
                enriched_pr['diff_size'] = diff_size
                current_batch.append(enriched_pr)
                successful += 1
            else:
                print(f"Failed to fetch diff for PR #{pr['html_url'].split('/')[-1]}")
                failed += 1

            processed += 1

            # Print progress every 10 PRs
            if processed % 10 == 0:
                print(f"\nProgress: {processed}/{total_prs} PRs processed")
                print(f"Success: {successful}, Failed: {failed}")
                print(f"Success rate: {(successful/processed)*100:.1f}%\n")

            # Save after each batch_size PRs
            if len(current_batch) >= batch_size:
                print(f"Saving batch of {len(current_batch)} PRs...")
                with open(output_file, 'w') as f:
                    json.dump(list(pr_lookup.values()), f, indent=2)
                current_batch = []

        except Exception as e:
            print(f"Error processing PR #{pr['html_url'].split('/')[-1]}: {e}")
            failed += 1
            continue

    # Save any remaining PRs
    if current_batch or failed > 0:  # Save if we have pending changes or failures
        print(f"Saving final batch...")
        with open(output_file, 'w') as f:
            json.dump(list(pr_lookup.values()), f, indent=2)

    # Print final stats
    print(f"\nFinal Results:")
    print(f"Total PRs processed: {processed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if processed > 0:
        print(f"Success rate: {(successful/processed)*100:.1f}%")
    print(f"Total PRs in system: {len(pr_lookup)}")

if __name__ == "__main__":
    #repo_name = "looker"
    repo_name = "nosara"
    #process_prs(repo_name, force_update=True)
    process_prs(repo_name, force_update=False)
