import os
import json
import time
from dotenv import load_dotenv
from plugins.memory.gravity import GravityMemoryProvider

# Cargar variables de entorno
load_dotenv()

def test_memory_lifecycle():
    print("==================================================")
    print("TEST DE CICLO DE VIDA DE MEMORIA (CRUD)")
    print("==================================================")

    provider = GravityMemoryProvider()
    # Forzar inicializacion manual
    # Inicializacion correcta
    provider.initialize(session_id="test-crud-session", user_id="test-crud-user")

    # 1. ADD
    print("\n[STEP 1] Agregando hecho inicial...")
    content_1 = "El usuario prefiere cafe solo sin azucar."
    res = provider.handle_manage_memory("add", content=content_1)
    print(f"Resultado: {res}")
    
    # El ID se genera por hash en la implementacion actual
    fact_id = "fact_" + str(hash(content_1))[:8]
    print(f"ID esperado: {fact_id}")

    time.sleep(2) # Esperar al hilo de fondo

    # 2. SEARCH
    print("\n[STEP 2] Verificando que existe...")
    res_search = provider.handle_manage_memory("search", content="cafe")
    print(f"Busqueda: {res_search}")

    # 3. UPDATE (Simulando lo que haria el agente)
    print("\n[STEP 3] Intentando actualizar (cambiando a cafe con leche)...")
    content_2 = "El usuario prefiere cafe con leche."
    # En la implementacion actual, esto podria fallar al sobreescribir si el ID cambia
    res_update = provider.handle_manage_memory("update", target_id=fact_id, content=content_2)
    print(f"Resultado Update: {res_update}")
    
    time.sleep(2)

    # 4. VERIFY DUPLICATION (Validando el bug sospechado)
    print("\n[STEP 4] Verificando si hay duplicados o si se actualizo...")
    res_final = provider.handle_manage_memory("search", content="cafe")
    print(f"Busqueda final: {res_final}")
    
    # 5. DELETE
    print("\n[STEP 5] Borrando el registro original...")
    res_delete = provider.handle_manage_memory("delete", target_id=fact_id)
    print(f"Resultado Delete: {res_delete}")

    print("\n==================================================")
    print("TEST FINALIZADO")
    print("==================================================")

if __name__ == "__main__":
    test_memory_lifecycle()
