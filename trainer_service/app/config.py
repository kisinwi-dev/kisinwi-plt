import os

class ServiceDomains:
    TASKER = f"http://{os.getenv('TASKER', 'localhost:6110')}"

config_domain = ServiceDomains()