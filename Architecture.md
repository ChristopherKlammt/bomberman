# Design-Entscheidungen

## Input für neuronales Netz
zu finden in ```callbacks.py``` unter ```state_to_features()```

Analog zu einem Bild mit verschiedenen Channels RGB werden als verschiedene Ebenen folgende Informationen aus dem game_state verwendet:
- Wände
- Crates
- Münzen
- Bomben
- Spieler

Diese werden gestackt und an das neuronale Netz übergeben.

## Auswählen der nächsten Aktion
zu finden in ```callbacks.py``` unter ```act()```

Hier wird im Spiel die nächste Aktion gewählt (basierend auf der Wahrscheinlichkeitsverteilung, die das neuronale Netz liefert).
Mit der Funktion ```get_valid_actions()``` werden nur mögliche valide Züge zurückgeliefert, sodass nicht versucht wird in die Wand zu laufen oder eine Bombe zu legen, wenn nicht möglich.

Im ersten Lernschritt wird die Möglichkeit eine Bombe zu legen komplett ausgeschaltet.

## Trainieren
```train.py```

In ```setup_training()``` wird alles initialisiert und das model erstellt (falls es nicht aus der Datei "my_agent.model" geladen werden konnte).

Nach jeder Runde wird ```game_events_occurred()``` aufgerufen und hier wird die Transition vom alten State in den neuen State mit der dazugehörigen Aktion sowie den Rewards der Liste von Transitions hinzugefügt.
Nach jedem Spiel wird ```end_of_round()``` aufgerufen und hier wird trainiert (Funktion ```train()``` aufgerufen).
Diese führt ein q-update target durch, sodass basierend auf den gemachten Erfahrungen gelernt wird.


Gesteuert werden kann die Belohnung durch ```reward_from_events()``` wo für die verschiedenen Events der reward festgelegt werden kann.
