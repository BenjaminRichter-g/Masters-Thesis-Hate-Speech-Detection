import requests
import sys

"""
Query a Mastodon instance for its federated peers using the public API.
"""
REQUEST_TIMEOUT = 10


def get_federated_peers(domain):
    """
    Fetch the list of instances federated by the given domain.
    Uses the Mastodon API endpoint /api/v1/instance/peers.
    """
    url = f'https://{domain}/api/v1/instance/peers'
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch peers from {url}: {e}")

    try:
        peers = resp.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON response from {url}")

    if not isinstance(peers, list):
        raise RuntimeError(f"Unexpected response format: expected list, got {type(peers).__name__}")

    return peers


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mastodon.instance>")
        sys.exit(1)

    domain = sys.argv[1].strip()
    try:
        peers = get_federated_peers(domain)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if peers:
        print(f"Instances federated by {domain} ({len(peers)}):")
        for peer in peers:
            print(f" - {peer}")
    else:
        print(f"No federated peers found for {domain}.")


if __name__ == '__main__':
    main()

