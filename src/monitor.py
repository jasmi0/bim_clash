"""
Real-Time Monitoring Module
Monitors BIM model for changes and detects new clashes
"""
import time
import hashlib
from typing import List, Dict, Any, Callable
from pathlib import Path
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor BIM model for changes and trigger clash detection"""
    
    def __init__(
        self,
        model_path: str,
        check_interval: int = 60,
        on_change_callback: Callable = None
    ):
        """
        Initialize Model Monitor
        
        Args:
            model_path: Path to the BIM model file
            check_interval: Interval in seconds to check for changes
            on_change_callback: Function to call when changes are detected
        """
        self.model_path = Path(model_path)
        self.check_interval = check_interval
        self.on_change_callback = on_change_callback
        self.last_hash = None
        self.last_check_time = None
        self.change_history = []
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start monitoring the model for changes"""
        self.is_monitoring = True
        self.last_hash = self._calculate_file_hash()
        self.last_check_time = datetime.now()
        
        logger.info(f"Started monitoring {self.model_path}")
        
        while self.is_monitoring:
            time.sleep(self.check_interval)
            self._check_for_changes()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("Stopped monitoring")
    
    def _check_for_changes(self):
        """Check if the model file has changed"""
        current_hash = self._calculate_file_hash()
        
        if current_hash != self.last_hash:
            logger.info("Model change detected")
            
            # Record change
            change_record = {
                'timestamp': datetime.now(),
                'previous_hash': self.last_hash,
                'new_hash': current_hash
            }
            self.change_history.append(change_record)
            
            # Update hash
            self.last_hash = current_hash
            self.last_check_time = datetime.now()
            
            # Trigger callback
            if self.on_change_callback:
                self.on_change_callback(change_record)
    
    def _calculate_file_hash(self) -> str:
        """Calculate hash of the model file"""
        if not self.model_path.exists():
            return None
        
        hash_md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_change_history(self) -> List[Dict]:
        """Get history of detected changes"""
        return self.change_history


class ClashMonitor:
    """Monitor and track clash status over time"""
    
    def __init__(self):
        """Initialize Clash Monitor"""
        self.clash_snapshots = []
        self.resolved_clashes = []
        self.new_clashes = []
    
    def add_snapshot(self, clashes: List, timestamp: datetime = None):
        """
        Add a snapshot of current clashes
        
        Args:
            clashes: List of detected clashes
            timestamp: Time of snapshot (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        snapshot = {
            'timestamp': timestamp,
            'clashes': clashes,
            'count': len(clashes),
            'by_severity': self._count_by_severity(clashes)
        }
        
        self.clash_snapshots.append(snapshot)
        
        # Compare with previous snapshot
        if len(self.clash_snapshots) > 1:
            self._compare_snapshots()
    
    def _count_by_severity(self, clashes: List) -> Dict[str, int]:
        """Count clashes by severity"""
        counts = {}
        for clash in clashes:
            severity = clash.severity
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _compare_snapshots(self):
        """Compare current snapshot with previous one"""
        if len(self.clash_snapshots) < 2:
            return
        
        current = self.clash_snapshots[-1]
        previous = self.clash_snapshots[-2]
        
        # Create sets of clash IDs
        current_ids = {
            f"{c.element1_guid}_{c.element2_guid}"
            for c in current['clashes']
        }
        previous_ids = {
            f"{c.element1_guid}_{c.element2_guid}"
            for c in previous['clashes']
        }
        
        # Find resolved and new clashes
        resolved_ids = previous_ids - current_ids
        new_ids = current_ids - previous_ids
        
        # Store resolved clashes
        for clash in previous['clashes']:
            clash_id = f"{clash.element1_guid}_{clash.element2_guid}"
            if clash_id in resolved_ids:
                self.resolved_clashes.append({
                    'clash': clash,
                    'resolved_at': current['timestamp']
                })
        
        # Store new clashes
        for clash in current['clashes']:
            clash_id = f"{clash.element1_guid}_{clash.element2_guid}"
            if clash_id in new_ids:
                self.new_clashes.append({
                    'clash': clash,
                    'detected_at': current['timestamp']
                })
        
        logger.info(f"Resolved clashes: {len(resolved_ids)}, New clashes: {len(new_ids)}")
    
    def get_trend_data(self) -> Dict[str, Any]:
        """Get trend data for visualization"""
        if not self.clash_snapshots:
            return {}
        
        timestamps = [s['timestamp'] for s in self.clash_snapshots]
        counts = [s['count'] for s in self.clash_snapshots]
        
        severity_trends = {}
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_trends[severity] = [
                s['by_severity'].get(severity, 0)
                for s in self.clash_snapshots
            ]
        
        return {
            'timestamps': timestamps,
            'total_counts': counts,
            'severity_trends': severity_trends,
            'resolved_count': len(self.resolved_clashes),
            'new_count': len(self.new_clashes)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.clash_snapshots:
            return {
                'total_snapshots': 0,
                'current_clashes': 0,
                'resolved_clashes': 0,
                'new_clashes': 0
            }
        
        current_snapshot = self.clash_snapshots[-1]
        
        return {
            'total_snapshots': len(self.clash_snapshots),
            'current_clashes': current_snapshot['count'],
            'resolved_clashes': len(self.resolved_clashes),
            'new_clashes': len(self.new_clashes),
            'first_check': self.clash_snapshots[0]['timestamp'],
            'last_check': current_snapshot['timestamp'],
            'current_severity_distribution': current_snapshot['by_severity']
        }
