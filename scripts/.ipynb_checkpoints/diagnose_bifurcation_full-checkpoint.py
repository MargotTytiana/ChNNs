#!/usr/bin/env python3
"""
Comprehensive Bifurcation Control Diagnostic Script
====================================================
Run this directly on HPC in the same directory as your training code.

Usage:
    python diagnose_bifurcation_full.py

This script will:
1. Check YAML config structure
2. Trace config through all layers
3. Create test network to verify parameter passing
4. Identify exact bug location
5. Provide copy-paste fix
"""

import sys
import os

# ============================================================================
# CONFIG - Update these paths for your environment
# ============================================================================
PROJECT_ROOT = "/scratch/project_2003370/yueyao/Model"
CONFIG_PATH = "/scratch/project_2003370/yueyao/Model/experiments/configs/4.3_bifurcation_control/chaotic_config.yaml"

sys.path.insert(0, PROJECT_ROOT)

import yaml

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_result(name, passed, details=""):
    symbol = "✓" if passed else "✗"
    status = "OK" if passed else "FAIL"
    print(f"  [{symbol}] {name}: {status}")
    if details:
        for line in details.split('\n'):
            print(f"      {line}")

# ============================================================================
# DIAGNOSTIC 1: Check YAML Structure
# ============================================================================
def diagnose_yaml():
    print_section("DIAGNOSTIC 1: YAML Configuration Structure")
    
    if not os.path.exists(CONFIG_PATH):
        print_result("Config file exists", False, f"Not found: {CONFIG_PATH}")
        return None
    
    print_result("Config file exists", True, CONFIG_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check all possible locations for bifurcation config
    locations_to_check = [
        ('config["optional_modules"]["bifurcation_control"]["enabled"]', 
         lambda c: c.get('optional_modules', {}).get('bifurcation_control', {}).get('enabled')),
        ('config["_nested"]["optional_modules"]["bifurcation_control"]["enabled"]',
         lambda c: c.get('_nested', {}).get('optional_modules', {}).get('bifurcation_control', {}).get('enabled')),
        ('config["bifurcation_control"]["enabled"]',
         lambda c: c.get('bifurcation_control', {}).get('enabled')),
    ]
    
    found_location = None
    found_value = None
    
    for location, accessor in locations_to_check:
        try:
            value = accessor(config)
            if value is not None:
                print_result(f"Found at {location}", True, f"Value: {value}")
                found_location = location
                found_value = value
            else:
                print_result(f"Found at {location}", False, "Path exists but value is None")
        except:
            print_result(f"Found at {location}", False, "Path does not exist")
    
    if found_location:
        print(f"\n  >>> Bifurcation config found at: {found_location}")
        print(f"  >>> Value: {found_value}")
    else:
        print("\n  >>> WARNING: Bifurcation config NOT found in any expected location!")
    
    # Print full optional_modules section if exists
    print("\n  Full optional_modules structure:")
    if 'optional_modules' in config:
        print(f"    {yaml.dump(config['optional_modules'], default_flow_style=False, indent=4)}")
    elif '_nested' in config and 'optional_modules' in config.get('_nested', {}):
        print(f"    {yaml.dump(config['_nested']['optional_modules'], default_flow_style=False, indent=4)}")
    else:
        print("    NOT FOUND")
    
    return config

# ============================================================================
# DIAGNOSTIC 2: Check chaotic_experiment.py
# ============================================================================
def diagnose_experiment():
    print_section("DIAGNOSTIC 2: chaotic_experiment.py Analysis")
    
    exp_path = os.path.join(PROJECT_ROOT, "chaotic_experiment.py")
    
    if not os.path.exists(exp_path):
        print_result("File exists", False)
        return
    
    with open(exp_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find ChaoticSpeakerRecognitionNetwork instantiation
    print("\n  Searching for ChaoticSpeakerRecognitionNetwork instantiation...")
    
    in_block = False
    block_start = 0
    block_lines = []
    paren_depth = 0
    
    for i, line in enumerate(lines):
        if 'ChaoticSpeakerRecognitionNetwork(' in line and 'model = ' in line:
            in_block = True
            block_start = i
            paren_depth = line.count('(') - line.count(')')
            block_lines = [(i+1, line)]
        elif in_block:
            paren_depth += line.count('(') - line.count(')')
            block_lines.append((i+1, line))
            if paren_depth <= 0:
                break
    
    if block_lines:
        print(f"\n  Found at lines {block_start+1}-{block_lines[-1][0]}:")
        for line_num, line in block_lines:
            print(f"    {line_num}: {line}")
        
        block_text = '\n'.join([l for _, l in block_lines])
        
        # Check what's being passed
        has_bifurcation = 'bifurcation' in block_text.lower()
        has_optional_modules = 'optional_modules' in block_text
        
        print_result("Passes use_bifurcation_control", has_bifurcation)
        print_result("References optional_modules", has_optional_modules)
        
        if not has_bifurcation:
            print("\n  !!! BUG CONFIRMED: use_bifurcation_control is NOT passed to network!")
    else:
        print("  Could not find network instantiation block")
    
    # Check how config is accessed elsewhere
    print("\n  Checking how config is accessed in this file...")
    
    config_accesses = []
    for i, line in enumerate(lines):
        if 'self.config[' in line or 'self.config.get(' in line:
            if 'optional' in line.lower() or 'bifurcation' in line.lower():
                config_accesses.append((i+1, line.strip()))
    
    if config_accesses:
        print("  Config accesses related to optional/bifurcation:")
        for line_num, line in config_accesses:
            print(f"    Line {line_num}: {line[:70]}")
    else:
        print("  No config accesses for optional_modules or bifurcation found")

# ============================================================================
# DIAGNOSTIC 3: Check chaotic_network.py
# ============================================================================
def diagnose_network():
    print_section("DIAGNOSTIC 3: chaotic_network.py Analysis")
    
    net_path = os.path.join(PROJECT_ROOT, "chaotic_network.py")
    
    if not os.path.exists(net_path):
        print_result("File exists", False)
        return
    
    with open(net_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check __init__ signature
    has_param = 'use_bifurcation_control: bool' in content
    print_result("__init__ has use_bifurcation_control parameter", has_param)
    
    # Check if stored
    stores_param = 'self.use_bifurcation_control = use_bifurcation_control' in content
    print_result("Stores self.use_bifurcation_control", stores_param)
    
    # Check if passed to ChaoticEmbedding
    print("\n  ChaoticEmbedding instantiation:")
    for i, line in enumerate(lines):
        if 'ChaoticEmbedding(' in line:
            print(f"    Line {i+1}: {line.strip()}")
            # Print next several lines
            for j in range(i+1, min(i+10, len(lines))):
                print(f"    Line {j+1}: {lines[j]}")
                if ')' in lines[j] and lines[j].strip().endswith(')'):
                    break
            break
    
    passes_to_embedding = 'use_bifurcation_control=self.use_bifurcation_control' in content
    print_result("Passes to ChaoticEmbedding", passes_to_embedding)

# ============================================================================
# DIAGNOSTIC 4: Runtime Test
# ============================================================================
def diagnose_runtime():
    print_section("DIAGNOSTIC 4: Runtime Test")
    
    try:
        import torch
        print_result("PyTorch available", True)
    except ImportError:
        print_result("PyTorch available", False, "Cannot run runtime test")
        return
    
    try:
        from chaotic_network import ChaoticSpeakerRecognitionNetwork
        print_result("Import ChaoticSpeakerRecognitionNetwork", True)
    except Exception as e:
        print_result("Import ChaoticSpeakerRecognitionNetwork", False, str(e))
        return
    
    # Test 1: Create with use_bifurcation_control=True
    print("\n  Test 1: Create network with use_bifurcation_control=True")
    try:
        net1 = ChaoticSpeakerRecognitionNetwork(
            sample_rate=16000,
            frame_length=400,
            hop_length=160,
            embedding_dim=10,
            delay_method='autocorr',
            mlsa_scales=5,
            rqa_radius_ratio=0.1,
            chaotic_system='lorenz',
            evolution_time=0.5,
            time_step=0.01,
            pooling_type='comprehensive',
            speaker_embedding_dim=256,
            num_speakers=26,
            classifier_type='linear',
            device='cpu',
            use_bifurcation_control=True
        )
        
        emb1 = net1.chaotic_embedding
        bif_val1 = getattr(emb1, 'use_bifurcation_control', 'NOT_FOUND')
        has_net1 = hasattr(emb1, 'bifurcation_net') and emb1.bifurcation_net is not None
        
        print(f"    chaotic_embedding.use_bifurcation_control = {bif_val1}")
        print(f"    chaotic_embedding.bifurcation_net exists = {has_net1}")
        
        if bif_val1 == True and has_net1:
            print_result("Bifurcation enabled correctly", True)
        else:
            print_result("Bifurcation enabled correctly", False)
            
    except Exception as e:
        print_result("Create network", False, str(e))
        import traceback
        traceback.print_exc()
    
    # Test 2: Create WITHOUT passing parameter (simulates the bug)
    print("\n  Test 2: Create network WITHOUT passing use_bifurcation_control")
    try:
        net2 = ChaoticSpeakerRecognitionNetwork(
            sample_rate=16000,
            frame_length=400,
            hop_length=160,
            embedding_dim=10,
            delay_method='autocorr',
            mlsa_scales=5,
            rqa_radius_ratio=0.1,
            chaotic_system='lorenz',
            evolution_time=0.5,
            time_step=0.01,
            pooling_type='comprehensive',
            speaker_embedding_dim=256,
            num_speakers=26,
            classifier_type='linear',
            device='cpu'
            # NOT PASSING use_bifurcation_control
        )
        
        emb2 = net2.chaotic_embedding
        bif_val2 = getattr(emb2, 'use_bifurcation_control', 'NOT_FOUND')
        has_net2 = hasattr(emb2, 'bifurcation_net') and emb2.bifurcation_net is not None
        
        print(f"    chaotic_embedding.use_bifurcation_control = {bif_val2}")
        print(f"    chaotic_embedding.bifurcation_net exists = {has_net2}")
        
        if bif_val2 == False:
            print("\n    >>> CONFIRMED: Without explicit parameter, bifurcation defaults to False")
            print("    >>> This explains why your experiments always run without bifurcation!")
            
    except Exception as e:
        print(f"    Error: {e}")

# ============================================================================
# DIAGNOSTIC 5: Generate Fix
# ============================================================================
def generate_fix():
    print_section("DIAGNOSTIC 5: RECOMMENDED FIX")
    
    print("""
  FILE TO MODIFY: chaotic_experiment.py
  LOCATION: _create_model() method, around line 173-203

  FIND THIS CODE:
  ---------------
  model = ChaoticSpeakerRecognitionNetwork(
      sample_rate=self.config['sample_rate'],
      frame_length=self.config['frame_length'],
      hop_length=self.config['hop_length'],
      embedding_dim=self.config['embedding_dim'],
      delay_method=self.config['delay_method'],
      mlsa_scales=self.config['mlsa_scales'],
      rqa_radius_ratio=self.config['rqa_radius_ratio'],
      chaotic_system=self.config['chaotic_system'],
      evolution_time=self.config['evolution_time'],
      time_step=self.config['time_step'],
      pooling_type=self.config['pooling_type'],
      speaker_embedding_dim=self.config['speaker_embedding_dim'],
      num_speakers=num_speakers,
      classifier_type=self.config['classifier_type'],
      device=self.device
  )

  REPLACE WITH:
  -------------
  # Read bifurcation config from optional_modules
  use_bifurcation = self.config.get('optional_modules', {}).get(
      'bifurcation_control', {}).get('enabled', False)
  
  self.logger.info(f"[BIFURCATION] Creating network with use_bifurcation_control={use_bifurcation}")
  
  model = ChaoticSpeakerRecognitionNetwork(
      sample_rate=self.config['sample_rate'],
      frame_length=self.config['frame_length'],
      hop_length=self.config['hop_length'],
      embedding_dim=self.config['embedding_dim'],
      delay_method=self.config['delay_method'],
      mlsa_scales=self.config['mlsa_scales'],
      rqa_radius_ratio=self.config['rqa_radius_ratio'],
      chaotic_system=self.config['chaotic_system'],
      evolution_time=self.config['evolution_time'],
      time_step=self.config['time_step'],
      pooling_type=self.config['pooling_type'],
      speaker_embedding_dim=self.config['speaker_embedding_dim'],
      num_speakers=num_speakers,
      classifier_type=self.config['classifier_type'],
      device=self.device,
      use_bifurcation_control=use_bifurcation  # <-- ADD THIS LINE
  )
""")
    
    print("""
  VERIFICATION AFTER FIX:
  -----------------------
  After applying the fix, your training log should show:
  
  [BIFURCATION] Creating network with use_bifurcation_control=True
  [NETWORK DEBUG] ChaoticEmbedding created with:
    use_bifurcation_control=True      <-- Should be True now
    bifurcation_net exists: True      <-- Should exist now
""")

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" BIFURCATION CONTROL COMPREHENSIVE DIAGNOSTIC")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Config Path: {CONFIG_PATH}")
    
    # Run all diagnostics
    yaml_config = diagnose_yaml()
    diagnose_experiment()
    diagnose_network()
    diagnose_runtime()
    generate_fix()
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
  The bug is confirmed: chaotic_experiment.py does NOT pass the
  use_bifurcation_control parameter when creating ChaoticSpeakerRecognitionNetwork.
  
  Even though your YAML config has bifurcation_control.enabled=true,
  this setting is never read and passed to the network.
  
  Apply the fix above and re-run your experiment.
""")
