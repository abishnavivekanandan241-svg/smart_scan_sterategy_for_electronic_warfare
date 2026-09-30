import pytest
import csv
import tempfile
import os
from src.preprocessing.tsrd_adapter import TSRDAdapter
from src.environment.pdw import PDW

def test_tsrd_adapter_mapping():
    adapter = TSRDAdapter()
    raw_row = {
        "UTCTime": "123.45",
        "RF": "1000.5",
        "PulseWidth": "2.5",
        "AOA": "45.0",
        "PA": "-30.0",
        "Class": "2"
    }
    
    pdw = adapter.adapt_record(raw_row)
    
    assert pdw.toa == 123.45
    assert pdw.centre_frequency == 1000.5
    assert pdw.pulse_width == 2.5
    assert pdw.aoa == 45.0
    assert pdw.amplitude == -30.0
    assert pdw.emitter_class == 2
    
def test_tsrd_adapter_missing_fields():
    adapter = TSRDAdapter()
    raw_row = {
        "UTCTime": "123.45",
        # RF is missing
        "PulseWidth": "invalid_number",
        "Class": "3.0"
    }
    
    pdw = adapter.adapt_record(raw_row)
    
    assert pdw.toa == 123.45
    assert pdw.centre_frequency == 0.0 # Default fallback
    assert pdw.pulse_width == 0.0      # Fallback for unparseable
    assert pdw.emitter_class == 3
    
def test_validate_data_and_conversion():
    adapter = TSRDAdapter()
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["UTCTime", "RF", "PulseWidth", "AOA", "PA", "Class"])
        writer.writerow(["1.0", "100", "2", "45", "-50", "1"])
        writer.writerow(["2.0", "200", "-1", "90", "-60", "2"]) # Non-positive PW
        writer.writerow(["3.0", "300", "1", "135", "-70", "1"])
        temp_path = f.name
        
    try:
        # Test Validation
        stats = adapter.validate_data(temp_path, expected_rows=3)
        assert stats["row_count"] == 3
        assert stats["expected_rows_matched"] is True
        assert stats["required_columns_exist"] is True
        assert stats["num_unique_classes"] == 2
        assert stats["rf_min"] == 100.0
        assert stats["rf_max"] == 300.0
        assert stats["non_positive_pw_count"] == 1
        
        # Test Conversion
        pdws = adapter.convert_to_pdws(temp_path)
        assert len(pdws) == 3
        assert pdws[0].toa == 1.0
        assert pdws[1].emitter_class == 2
        
        # Test Limit
        pdws_limited = adapter.convert_to_pdws(temp_path, limit=2)
        assert len(pdws_limited) == 2
        
    finally:
        os.remove(temp_path)
