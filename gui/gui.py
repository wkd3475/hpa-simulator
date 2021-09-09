class SimApp(QWidget):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('pods_min:'), 0, 0)
        grid.addWidget(QLabel('pods_max:'), 1, 0)
        grid.addWidget(QLabel('timeout (sec):'), 2, 0)
        grid.addWidget(QLabel('autoscale period (sec):'), 3, 0)
        grid.addWidget(QLabel('simulation period (sec):'), 4, 0)
        grid.addWidget(QLabel('readiness_probe (sec):'), 5, 0)
        grid.addWidget(QLabel('scaling_tolerance:'), 6, 0)
        btn_run_one = QPushButton('Run One Step', self)
        grid.addWidget(btn_run_one, 7, 0)

        setting = self.env.get_setting()

        grid.addWidget(QLabel(str(setting["pods_min"])), 0, 1)
        grid.addWidget(QLabel(str(setting["pods_max"])), 1, 1)
        grid.addWidget(QLabel(str(setting["timeout"])), 2, 1)
        grid.addWidget(QLabel(str(setting["autoscale_period"])), 3, 1)
        grid.addWidget(QLabel(str(setting["simulation_period"])), 4, 1)
        grid.addWidget(QLabel(str(setting["readiness_probe"])), 5, 1)
        grid.addWidget(QLabel(str(setting["scaling_tolerance"])), 6, 1)
        btn_run = QPushButton('Run', self)
        grid.addWidget(btn_run, 7, 1)

        self.setWindowTitle('Simulator')
        self.show()

    def run_one_step(self):
        self.env.next_rl_state()


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.loadHistroy()
        self.initUI()

    def loadHistroy(self):
        self.traffic_history = []
        f = open('./data/nasa-http-data-3.csv', 'r', encoding='utf-8')
        rdr =csv.reader(f)
        j = 0
        for line in rdr:
            if j == 0:
                j = j+1
                self.traffic_history.append(0)
                continue

            # for i in range(60):
            self.traffic_history.append(int(line[0]))

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('pods_min:'), 0, 0)
        grid.addWidget(QLabel('pods_max:'), 1, 0)
        grid.addWidget(QLabel('timeout (sec):'), 2, 0)
        grid.addWidget(QLabel('autoscale period (sec):'), 3, 0)
        grid.addWidget(QLabel('simulation period (sec):'), 4, 0)
        grid.addWidget(QLabel('readiness_probe (sec):'), 5, 0)
        grid.addWidget(QLabel('scaling_tolerance:'), 6, 0)

        self.qe_pods_min = QLineEdit()
        self.qe_pods_min.setText("1")
        self.qe_pods_max = QLineEdit()
        self.qe_pods_max.setText("6")
        self.qe_timeout = QLineEdit()
        self.qe_timeout.setText("0.3")
        self.qe_autoscale_period = QLineEdit()
        self.qe_autoscale_period.setText("15")
        self.qe_simulation_period = QLineEdit()
        self.qe_simulation_period.setText("0.1")
        self.qe_readiness_probe = QLineEdit()
        self.qe_readiness_probe.setText("3")
        self.qe_scaling_tolerance = QLineEdit()
        self.qe_scaling_tolerance.setText("0.1")

        grid.addWidget(self.qe_pods_min, 0, 1)
        grid.addWidget(self.qe_pods_max, 1, 1)
        grid.addWidget(self.qe_timeout, 2, 1)
        grid.addWidget(self.qe_autoscale_period, 3, 1)
        grid.addWidget(self.qe_simulation_period, 4, 1)
        grid.addWidget(self.qe_readiness_probe, 5, 1)
        grid.addWidget(self.qe_scaling_tolerance, 6, 1)

        btn_set = QPushButton('SetButton', self)
        btn_set.clicked.connect(self.set_env)
        grid.addWidget(btn_set, 7, 1)

        self.setWindowTitle('Setting')
        self.show()

    def set_env(self):
        self.env = Environment(
            application_profile=application_profile, 
            traffic_history=self.traffic_history, 
            pods_min=int(self.qe_pods_min.text()), 
            pods_max=int(self.qe_pods_max.text()), 
            timeout=float(self.qe_timeout.text()), 
            autoscale_period=float(self.qe_autoscale_period.text()), 
            simulation_period=float(self.qe_simulation_period.text()), 
            readiness_probe=float(self.qe_readiness_probe.text()), 
            scaling_tolerance=float(self.qe_scaling_tolerance.text())
        )
        self.sim = SimApp(self.env)
        self.sim.show()
        self.hide()

# app = QApplication(sys.argv)
# ex = MyApp()
# sys.exit(app.exec_())