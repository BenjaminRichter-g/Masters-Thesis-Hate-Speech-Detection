import subprocess

INPUT_FILE = 'racism_domains.txt'
VALID_OUTPUT = 'valid_instances.txt'
TIMELINE_FILE = 'output_timeline.jsonl'
TIMEOUT_SECONDS = 1

LOG_CRAWLED = 'Crawled 1 statuses'
LOG_SAVED = f'Output saved to {TIMELINE_FILE}'


def main():
    successes = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]

    for domain in domains:
        
        if "*" in domain:
            print(f'Skipping wildcard domain: {domain}')
            continue

        print(f'Testing {domain}...', end=' ')
        try:
            proc = subprocess.run(
                [
                    'mastodoner', 'instance',
                    '--instance-url', domain,
                    '--timeline', TIMELINE_FILE,
                    '--limit', '1'
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            print('TIMEOUT')
            continue
        except subprocess.CalledProcessError:
            print('FAIL')
            continue

        logs = proc.stdout + proc.stderr
        if LOG_CRAWLED in logs and LOG_SAVED in logs:
            print('OK')
            successes.append(domain)
        else:
            print('FAIL')

    with open(VALID_OUTPUT, 'w', encoding='utf-8') as f:
        for d in successes:
            f.write(d + '\n')

    print(f"Done. Valid instances listed in {VALID_OUTPUT}.")


if __name__ == '__main__':
    main()
