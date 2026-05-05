# PRD: Strategic Model Routing & Hybrid Memory Integration

## Problem Statement

El usuario enfrenta limitaciones de cuota y rendimiento subóptimo al utilizar un único modelo de Gemini para todas las tareas del agente (chat, codificación, extracción de hechos, visión). Diferentes tareas tienen distintos requisitos de latencia, profundidad de razonamiento y modalidad. Depender de un modelo predeterminado global hace que las "tareas secundarias" (llamadas auxiliares) fallen por cuota o sean innecesariamente lentas para operaciones simples.

## Solution

Implementar un sistema de enrutamiento estratégico de modelos para operaciones auxiliares que sea consciente de la tarea. Mapear tareas específicas (Extracción de Hechos, Reflexión, Codificación, Visión, Planificación, Validación) al modelo más adecuado de la familia Gemini/Gemma disponible. Este sistema se integra con "Gravity Hybrid Memory" (Pinecone + Supabase) para garantizar que las tareas auxiliares operen sobre una base de memoria semántica y estructurada eficiente.

## User Stories

1. Como desarrollador, quiero que las tareas de codificación auxiliar utilicen el modelo más potente disponible (Gemma 4 31B), para que las correcciones y análisis técnicos sean de alta fidelidad.
2. Como usuario, quiero que la extracción de hechos y resúmenes de memoria utilicen modelos de baja latencia (Gemini 3.1 Flash Lite), para que el agente responda más rápido sin agotar la cuota de modelos premium.
3. Como agente, quiero que el enrutador estratégico elija automáticamente el mejor modelo para validación de planes (Gemma 3 12B), de modo que pueda verificar mis pasos internamente de forma económica.
4. Como arquitecto, quiero que el sistema de memoria híbrida combine Pinecone (semántico) y Supabase (estructurado) para que el contexto recuperado sea preciso y esté disponible para todas las tareas auxiliares.
5. Como usuario con cuota limitada, quiero que el sistema evite el uso de modelos no disponibles (como Flash 3 Preview) y use variantes de Gemma con mayor disponibilidad.

## Implementation Decisions

- **Módulo de Enrutamiento**: Modificación de `agent/auxiliary_client.py` para interceptar llamadas cuando el proveedor principal es Google/Gemini.
- **Registro Estratégico**: Creación de `_GEMINI_STRATEGIC_MODELS` para mapear tipos de tareas a modelos específicos (e.g., `coding` -> `gemma-4-31b-it`, `vision` -> `gemma-4-26b-a4b-it`).
- **Integración de Memoria**: Uso de la clase `GravityMemoryProvider` que orquestra búsquedas vectoriales en Pinecone y almacenamiento relacional en Supabase.
- **Fallback Dinámico**: Definición de un modelo por defecto robusto (`gemma-4-31b-it`) para cualquier tarea auxiliar no especificada.
- **Prioridad de Configuración**: El sistema respeta las anulaciones manuales en `config.yaml`, permitiendo que el usuario o el agente ajusten modelos específicos por tarea.

## Testing Decisions

- **Pruebas de Enrutamiento**: Verificar que al ejecutar una tarea de "coding", los logs de `auxiliary_client` muestren la selección de `gemma-4-31b-it`.
- **Validación de Memoria**: Confirmar que las llamadas auxiliares de extracción de hechos persistan correctamente en Supabase y sean recuperables mediante la búsqueda semántica de Pinecone.
- **Simulación de Cuota**: Asegurar que el cambio del modelo predeterminado de Gemini a Flash Lite para tareas ligeras reduce la presión sobre los límites de la API de Google.

## Out of Scope

- Implementación de enrutamiento estratégico para proveedores distintos a Google/Gemini en esta fase.
- Modificación de la lógica principal de chat (que sigue usando el modelo configurado por el usuario).

## Further Notes

Este sistema sienta las bases para un "Agente Autónomo de Bajo Costo" que maximiza el uso de modelos Gemma gratuitos o de alta cuota para procesos internos, reservando la inteligencia de Gemini para la interacción directa con el usuario.
