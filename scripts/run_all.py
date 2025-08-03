import sys
import subprocess
import time


def run_test(test_file):
    """Ejecuta un test y retorna el resultado"""
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd='./')
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def format_metric_line(line):
    """Formatea una línea de métrica para mostrar solo 4 decimales"""
    import re
    # Buscar números con muchos decimales y formatearlos
    def replace_number(match):
        number = float(match.group())
        return f"{number:.4f}"
    
    # Regex para encontrar números decimales largos
    formatted_line = re.sub(r'\d+\.\d{5,}', replace_number, line)
    return formatted_line

def extract_key_results(output, test_type):
    """Extrae solo los resultados importantes del output"""
    lines = output.strip().split('\n')
    
    if test_type == "linear":
        # Buscar resultados finales de MSE
        sklearn_section = False
        my_model_section = False
        results = []
        
        for line in lines:
            if "=== SKLEARN ===" in line:
                sklearn_section = True
                results.append("📊 SKLEARN RESULTS:")
                continue
            elif "=== MI MODELO ===" in line:
                sklearn_section = False
                my_model_section = True
                results.append("🎯 MI MODELO RESULTS:")
                continue
            
            if sklearn_section or my_model_section:
                if "MSE" in line:
                    formatted_line = format_metric_line(line.strip())
                    results.append(f"   {formatted_line}")
        
        return "\n".join(results)
    
    elif test_type == "logistic":
        # Buscar métricas finales
        sklearn_section = False
        my_model_section = False
        results = []
        
        for line in lines:
            if "=== SKLEARN ===" in line:
                sklearn_section = True
                results.append("📊 SKLEARN RESULTS:")
                continue
            elif "=== MI MODELO ===" in line:
                sklearn_section = False
                my_model_section = True
                results.append("🎯 MI MODELO RESULTS:")
                continue
            
            if sklearn_section or my_model_section:
                if any(metric in line for metric in ["Accuracy:", "Precision:", "Recall:", "F1:"]):
                    formatted_line = format_metric_line(line.strip())
                    results.append(f"   {formatted_line}")
        
        return "\n".join(results)
    
    return output

def main():
    print("=" * 50)
    print("🤖 Mi implementacion de la Regression")
    print("=" * 50)
    print("🚀 Ejecutando tests...")
    print()
    
    tests = [
        ("tests/test_linear.py", "📈 Linear Regression", "linear"),
        ("tests/test_logistic.py", "🎯 Logistic Regression", "logistic")
    ]
    
    all_passed = True
    results_summary = []
    
    for test_file, test_name, test_type in tests:
        print(f"⏳ Ejecutando {test_name}...", end=" ")
        start_time = time.time()
        
        success, output, error = run_test(test_file)
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"✅ COMPLETADO ({elapsed:.1f}s)")
            
            # Extraer solo resultados clave
            key_results = extract_key_results(output, test_type)
            results_summary.append(f"\n🔹 {test_name}")
            results_summary.append(key_results)
            
        else:
            print(f"❌ Ha habido un error")
            print(f"Error: {error}")
            all_passed = False
    
    # Mostrar resumen limpio
    print("\n" + "=" * 50)
    print("📈 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    for result in results_summary:
        print(result)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
    else:
        print("⚠️  Algunos tests fallaron")
    
    print("=" * 50)

if __name__ == '__main__':
    main()