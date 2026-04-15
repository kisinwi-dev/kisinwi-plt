import os

class ServiceDomains:
    TASKER = f"http://{os.getenv('TASKER_DOMAIN', 'localhost:6110')}"
    METRIC = f"http://{os.getenv("METRICS_DOMAIN", "localhost:6310")}"

config_domain = ServiceDomains()