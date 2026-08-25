"""Gerador de mensagens que fala o contrato UDP/JSON real do tracker
(io_/receiver.py + io_/parser.py) - representa as estacoes externas para
testar o pipeline completo (rede -> parsing -> buffer -> associacao ->
fusao -> tracking) sem depender das estacoes reais.

Diferente de `simulation/` (o simulador CIENTIFICO/offline, que chama
`Tracker.process_batch` diretamente, em lockstep, para os experimentos
metodologicos e testes de avaliacao): este pacote representa estacoes de
rede de verdade, cada uma na sua propria frequencia,
enviando JSON por UDP de verdade. Reusa `simulation.virtual_station` e
`simulation.noise` para a parte de ruido/identidade local - a unica coisa
nova aqui e a camada de rede, cronograma independente por estacao e
serializacao para o contrato de fio.

Nome deliberadamente diferente de `simulation/` para nao dar a entender que
sao a mesma coisa (ver readme.md)."""
