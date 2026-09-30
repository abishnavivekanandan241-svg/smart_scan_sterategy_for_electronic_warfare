import csv
from typing import List, Dict, Any, Tuple
from src.environment.pdw import PDW

class TSRDAdapter:
    """
    Adapter to convert external TSRD CSV data into internal PDW format.
    """
    
    def __init__(self):
        pass
        
    def adapt_record(self, raw_record: Dict[str, str]) -> PDW:
        """
        Converts a single TSRD CSV row (as strings) into a 5-D PDW.
        Expected mapping:
        UTCTime -> ToA
        RF -> Centre Frequency
        PulseWidth -> Pulse Width
        AOA -> AoA
        PA -> Amplitude
        Class -> emitter_class (ground truth metadata)
        """
        # Parse fields, defaulting to 0.0 if missing/unparseable for robustness
        try:
            toa = float(raw_record.get("UTCTime", 0.0))
        except ValueError:
            toa = 0.0
            
        try:
            cf = float(raw_record.get("RF", 0.0))
        except ValueError:
            cf = 0.0
            
        try:
            pw = float(raw_record.get("PulseWidth", 0.0))
        except ValueError:
            pw = 0.0
            
        try:
            aoa = float(raw_record.get("AOA", 0.0))
        except ValueError:
            aoa = 0.0
            
        try:
            amp = float(raw_record.get("PA", 0.0))
        except ValueError:
            amp = 0.0
            
        try:
            e_class = int(float(raw_record.get("Class", 0)))
        except ValueError:
            e_class = None
            
        return PDW(
            toa=toa,
            centre_frequency=cf,
            pulse_width=pw,
            aoa=aoa,
            amplitude=amp,
            emitter_class=e_class
        )
        
    def validate_data(self, filepath: str, expected_rows: int | None = None) -> Dict[str, Any]:
        """
        Validates the TSRD dataset file in a streaming fashion.
        """
        stats = {
            "row_count": 0,
            "required_columns_exist": False,
            "missing_values_count": 0,
            "unique_classes": set(),
            "rf_min": float('inf'),
            "rf_max": float('-inf'),
            "aoa_min": float('inf'),
            "aoa_max": float('-inf'),
            "non_positive_pw_count": 0,
            "expected_rows_matched": None
        }
        
        required_cols = {"UTCTime", "RF", "PulseWidth", "AOA", "PA", "Class"}
        
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames:
                actual_cols = set(reader.fieldnames)
                stats["required_columns_exist"] = required_cols.issubset(actual_cols)
                
            for row in reader:
                stats["row_count"] += 1
                
                # Check for missing values (empty strings)
                if any(v.strip() == "" for v in row.values()):
                    stats["missing_values_count"] += 1
                    
                # Parse numeric values for stats
                try:
                    c = int(float(row.get("Class", -1)))
                    stats["unique_classes"].add(c)
                except ValueError:
                    pass
                    
                try:
                    rf = float(row.get("RF", 0.0))
                    stats["rf_min"] = min(stats["rf_min"], rf)
                    stats["rf_max"] = max(stats["rf_max"], rf)
                except ValueError:
                    pass
                    
                try:
                    aoa = float(row.get("AOA", 0.0))
                    stats["aoa_min"] = min(stats["aoa_min"], aoa)
                    stats["aoa_max"] = max(stats["aoa_max"], aoa)
                except ValueError:
                    pass
                    
                try:
                    pw = float(row.get("PulseWidth", 0.0))
                    if pw <= 0:
                        stats["non_positive_pw_count"] += 1
                except ValueError:
                    stats["non_positive_pw_count"] += 1
                    
        if expected_rows is not None:
            stats["expected_rows_matched"] = (stats["row_count"] == expected_rows)
            
        # Convert set to count for the final report
        stats["num_unique_classes"] = len(stats["unique_classes"])
        stats.pop("unique_classes")
        
        return stats
        
    def convert_to_pdws(self, filepath: str, limit: int | None = None) -> List[PDW]:
        """
        Loads the TSRD CSV and converts rows to PDW objects.
        """
        pdws = []
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                pdws.append(self.adapt_record(row))
        return pdws
