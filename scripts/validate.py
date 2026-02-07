#!/usr/bin/env python3
"""
ShelfSense AI - Code Validation Script

Performs static analysis to identify potential issues:
- Syntax errors
- Import inconsistencies 
- Missing definitions
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict

def check_syntax(filepath: Path) -> list[str]:
    """Check Python file for syntax errors."""
    errors = []
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Syntax error in {filepath}: {e}")
    return errors

def extract_definitions(filepath: Path) -> dict:
    """Extract class, function, and module-level variable definitions from a file."""
    definitions = {"classes": set(), "functions": set(), "variables": set()}
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.iter_child_nodes(tree):  # Only top-level nodes
            if isinstance(node, ast.ClassDef):
                definitions["classes"].add(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                definitions["functions"].add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions["variables"].add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                definitions["variables"].add(node.target.id)
    except:
        pass
    return definitions

def extract_imports(filepath: Path) -> list[tuple]:
    """Extract imports from a file."""
    imports = []
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append((module, alias.name))
    except:
        pass
    return imports

def validate_project(root: Path) -> dict:
    """Validate the project structure and code."""
    results = {
        "syntax_errors": [],
        "import_issues": [],
        "file_count": 0,
        "class_count": 0,
        "function_count": 0,
    }
    
    # Collect all definitions
    all_definitions = {}
    shelfsense_dir = root / "shelfsense"
    
    for py_file in shelfsense_dir.rglob("*.py"):
        results["file_count"] += 1
        
        # Check syntax
        errors = check_syntax(py_file)
        results["syntax_errors"].extend(errors)
        
        # Extract definitions
        defs = extract_definitions(py_file)
        results["class_count"] += len(defs["classes"])
        results["function_count"] += len(defs["functions"])
        
        rel_path = py_file.relative_to(root)
        module_path = str(rel_path).replace("/", ".").replace(".py", "")
        all_definitions[module_path] = defs
    
    # Check imports in __init__.py files
    for init_file in shelfsense_dir.rglob("__init__.py"):
        imports = extract_imports(init_file)
        rel_path = init_file.relative_to(root)
        
        for module, name in imports:
            if module.startswith("shelfsense."):
                # Check if the import target exists
                target_module = module
                expected_module = target_module.replace(".", "/") + ".py"
                expected_path = root / expected_module
                
                if expected_path.exists():
                    defs = all_definitions.get(target_module, {"classes": set(), "functions": set(), "variables": set()})
                    all_names = defs["classes"] | defs["functions"] | defs["variables"]
                    if name not in all_names:
                        results["import_issues"].append(
                            f"{rel_path}: '{name}' not found in {module}"
                        )
    
    return results

def main():
    root = Path("/home/claude/shelfsense-ai")
    
    print("=" * 60)
    print("ShelfSense AI - Code Validation Report")
    print("=" * 60)
    
    results = validate_project(root)
    
    print(f"\n📊 Statistics:")
    print(f"   Python files: {results['file_count']}")
    print(f"   Classes: {results['class_count']}")
    print(f"   Functions: {results['function_count']}")
    
    print(f"\n✅ Syntax Errors: {len(results['syntax_errors'])}")
    for err in results['syntax_errors'][:5]:
        print(f"   - {err}")
    
    print(f"\n⚠️  Import Issues: {len(results['import_issues'])}")
    for issue in results['import_issues'][:10]:
        print(f"   - {issue}")
    
    if results['import_issues']:
        print(f"\n   ... and {max(0, len(results['import_issues']) - 10)} more")
    
    print("\n" + "=" * 60)
    if results['syntax_errors'] or results['import_issues']:
        print("❌ Validation found issues that need fixing")
        return 1
    else:
        print("✅ Basic validation passed")
        return 0

if __name__ == "__main__":
    sys.exit(main())
