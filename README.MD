Hallo Sascha!

Der Code funktioniert wie folgt. Wenn man von Beginn an starten möchte, fängt man mit dem Script: "0_create_training_samples.py" an.
Das erschafft den Trainingsdatensatz für das neuronale Netzwerk. Dabei werden mehrere Millionen zufällige state-transitions im Environment aufgezeichnet. Daraus kann das neuronale Netzwerk dann das Verhalten des Systems lernen.
Danach muss mithilfe der Trainingsdaten das neuronale Netzwerk trainiert werden. "1_fit_network.py" führt das Training durch und speichert die Gewichte des Netzwerks als "weights.h5".
Im Anschluss führt man mit "2_evaluation.py" die Berechnung der Ergebnisse durch. Hier findet dann auch die Monte-Carlo tree search statt.
In Zukunft könnte mit dem "main.py" Script, das Netzwerk feingetunt werden. Dabei werden dann nicht mehr zufällige state transitions gesammelt, sondern der Algorithmus lernt aus seinem eigenem Verhalten und trainiert, während er weiter erforscht.

Alle Einstellungen des Netzwerks, sowie die Suchparameter sind in der Config-Klasse voreingestellt. Diese ist in "helper.py" enthalten.
Die Samples habe ich aus dem Upload gelöscht, da sie fast 2GB einnehmen. Die Gewichte des Netzwerks sind aber in dem Ordner enthalten.
Da es sich um einen älteren Stand der VPPlib handelt, fehlt hier noch der Input_Houses Ordner. Dieser ist ebenfalls zu groß, um hier hochgeladen zu werden. Ich habe ihn nochmal für dich hier abgespeichert:

https://drive.google.com/drive/folders/1qpYdOcQBPMay6_VnAu8_jhxQTr5dPh-8?usp=sharing

Was in Zukunft zu tun wäre:
 - Einbindung der aktuellen VPPlib und Möglichkeiten finden, dass das Environment nicht auf eigene Funktionen angewiesen ist. Als native Einbindung in die VPPlib ist das Environment noch nicht geeignet. Das ist aber durchaus machbar.
 - Bessere Neuronale Netzwerk Architekturen finden, die in der Lage sind nicht nur die Reward, sondern auch den nächsten State vorherzusagen. Damit könnte man das Environment nahezu überflüssig machen und der Algorithmus wäre in der Lage nur aus den Beobachtungen eines Energiesystems zu lernen! Dann wäre es nicht mehr zwingend nötig, das virtuelle Kraftwerk vorher zu modellieren!
 - Wir verwenden noch "perfect foresight". Ich habe aber ein paar Ideen, auch statistische Faktoren, bspw. über die erwartete PV-Leistung zu integrieren.
 - Das Modul "prioritized_experience_sampler.py" wird aktuell nicht genutzt, könnte aber sobald das Netzwerk aus eigenen Erfahrungen trainiert wird sehr nützlich sein. Es trifft eine Vorauswahl über die Trainings-Samples je Trainingsschritt, damit vor allem die wichtigen state-transitions für das Training genutzt werden.
