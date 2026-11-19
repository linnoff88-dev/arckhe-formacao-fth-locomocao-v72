# Plano de treino de locomoção

Robô: Unitree G1 29 GDL, unitree_g1/scene.xml do Menagerie 62ab00f
Comando: vx de -0.5 a 1.0 m/s, vy de -0.3 a 0.3 m/s, giro de -0.2 a 0.2 rad/s
Recompensa: track_linear_velocity 2.0, track_angular_velocity 2.0, upright 1.0, pose 1.0, action_rate_l2 -0.1, is_alive 5.0
Currículo: plano → degrau de 3 cm → degrau de 6 cm → rampa de 10° → pedriscos
Avaliação: 18/20 episódios de pé com empurrão de 1.0 m/s, erro de velocidade abaixo de 0.1 m/s
