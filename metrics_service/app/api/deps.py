from app.core.storage import CVMetricManager

manager = CVMetricManager()

def get_metrics_manager():
    try:
        manager.connect()
        yield manager
    finally:
        manager.disconnect()