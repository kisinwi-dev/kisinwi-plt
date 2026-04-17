from app.core.storage import CVMetricManager

def get_metrics_manager():
    manager = CVMetricManager()
    try:
        manager.connect()
        yield manager
    finally:
        manager.disconnect()