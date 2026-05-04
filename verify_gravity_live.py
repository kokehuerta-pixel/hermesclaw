import os
import json
import time
import logging
from dotenv import load_dotenv

# Configurar logging para ver errores internos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_gravity")

# Cargar .env
load_dotenv()

# Asegurarse de que el directorio actual está en el path para importar plugins
import sys
sys.path.append(os.getcwd())

try:
    from plugins.memory.gravity import GravityMemoryProvider
except ImportError as e:
    print(f"❌ Error al importar GravityMemoryProvider: {e}")
    print("Asegúrate de ejecutar este script desde la raíz del proyecto.")
    sys.exit(1)

def verify_live():
    print("\n" + "="*50)
    print("--- VERIFICACIÓN DEL SISTEMA DE MEMORIA GRAVITY ---")
    print("="*50 + "\n")
    
    provider = GravityMemoryProvider()
    
    # 0. Verificar disponibilidad
    if not provider.is_available():
        print("❌ Error: Las dependencias o credenciales no están completas.")
        print(f"SUPABASE_URL: {'OK' if os.getenv('SUPABASE_URL') else 'MISSING'}")
        print(f"PINECONE_API_KEY: {'OK' if os.getenv('PINECONE_API_KEY') else 'MISSING'}")
        print(f"PINECONE_INDEX_NAME: {'OK' if os.getenv('PINECONE_INDEX_NAME') else 'MISSING'}")
        return

    # Simular inicialización
    session_id = f"test-session-{int(time.time())}"
    provider.initialize(session_id=session_id, user_id="test-user")
    
    if not provider._initialized:
        print("[FAIL] Error: El proveedor no se inicializo correctamente.")
        return

    print("[SUCCESS] Proveedor inicializado con exito.")
    print(f"Session ID: {session_id}")

    # 1. Probar escritura manual (manage_memory)
    print("\n[TEST 1] Escritura manual (manage_memory)...")
    fact_content = f"El usuario prefiere trabajar en proyectos de IA (Test ID: {session_id})"
    res_json = provider.handle_manage_memory(
        action="add", 
        content=fact_content, 
        metadata={"sub_category": "user_preference"}
    )
    res = json.loads(res_json)
    if res.get("success"):
        print(f"[SUCCESS] Hecho guardado en Supabase y Pinecone.")
    else:
        print(f"[FAIL] Error al guardar hecho: {res.get('error')}")

    # 2. Probar búsqueda semántica (prefetch)
    print("\n[TEST 2] Busqueda semantica (prefetch)...")
    print("Esperando 3 segundos para propagacion en Pinecone...")
    time.sleep(3) 
    
    context = provider.prefetch("Cuales son las preferencias del usuario respecto a proyectos?")
    if "IA" in context:
        print("[SUCCESS] Busqueda semantica exitosa.")
        print(f"Fragmento recuperado: {context.strip()[:150]}...")
    else:
        print("[WARN] Advertencia: No se recupero el hecho esperado inmediatamente.")
        print("Esto es normal debido a la latencia eventual de los indices vectoriales.")
        print(f"Contexto recibido: {context}")

    # 3. Probar extracción autónoma (sync_turn)
    print("\n[TEST 3] Extraccion autonoma de hechos (sync_turn)...")
    user_msg = "Por cierto, mi lenguaje de programacion favorito es Rust."
    assistant_msg = "Que interesante! Guardare en tu perfil que prefieres Rust."
    
    print("Simulando turno de conversacion...")
    provider.sync_turn(user_msg, assistant_msg)
    
    print("Esperando 12 segundos para que el modelo auxiliar extraiga el hecho en segundo plano...")
    # El sync_turn lanza un hilo que llama a Gemini para extraer hechos y luego los guarda.
    for i in range(12, 0, -1):
        print(f"Esperando... {i}s", end="\r")
        time.sleep(1)
    print("\n")
    
    # 4. Verificar si se extrajo el lenguaje favorito
    print("[TEST 4] Verificando extraccion autonoma...")
    context_rust = provider.prefetch("Cual es el lenguaje favorito del usuario?")
    if "Rust" in context_rust:
        print("[SUCCESS] EXITOTOTAL! El sistema extrajo el hecho 'Rust' de la conversacion automaticamente.")
        print(f"Contexto recuperado:\n{context_rust}")
    else:
        print("[WARN] El hecho 'Rust' no aparece todavia en la memoria semantica.")
        print("Puede ser por latencia o porque el modelo auxiliar decidio que no era un hecho permanente.")
        print(f"Contexto actual:\n{context_rust if context_rust else '(Vacio)'}")

    print("\n" + "="*50)
    print("VERIFICACIÓN FINALIZADA")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_live()
