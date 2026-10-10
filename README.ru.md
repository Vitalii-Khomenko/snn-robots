# StackChan: нейроморфный мозг

Прошивка для настольного робота **M5Stack CoreS3 StackChan**. Всё поведение рождается в гибридном мозге на самом
ESP32-S3: маленькие сети зрения и слуха, **импульсное ядро из 768 нейронов** (внимание, новизна, скука, любопытство,
сон), рекуррентная политика, которая выбирает действия и одно из 21 лиц, и аналоговый «дроид-голос», который
составляет сеть. Ни деревьев поведения, ни готовых анимаций и звуков.

В проекте есть симулятор («цифровой двойник»: настоящий код прошивки в виртуальной комнате) и весь конвейер обучения
каждой сети на ПК.

С чего начать: [stackchan-neuromorphic/README.md](stackchan-neuromorphic/README.md) (раздел «Getting started»),
затем [ARCHITECTURE](stackchan-neuromorphic/docs/ARCHITECTURE.md), [HARDWARE](stackchan-neuromorphic/docs/HARDWARE.md),
[DEVELOPMENT](stackchan-neuromorphic/docs/DEVELOPMENT.md). Документация на английском.

Файлы `TODO_LATER.txt`, `RECORDING_PLAN.txt`, `VISION_PLAN.txt` и `Ideas-Tmp/` - личные рабочие заметки владельца
(по-русски), они нужны только как история и план.

Лицензия: MIT. Пароль Wi-Fi и IP устройств владельца в репозиторий не входят: шаблоны лежат в `secrets/*.example.h`.
