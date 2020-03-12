import time
from collections import deque
from helper import ActionSpaceGenerator
import numpy as np
import pandas as pd

from model.battery_electric_vehicle import battery_electric_vehicle
from model.electrical_energy_storage import electrical_energy_storage
from model.heat_pump import heat_pump
from model.photovoltaic import photovoltaic
from model.thermal_energy_storage import thermal_energy_storage
from model.user_profile import user_profile


LOG_LEN = 95
log_global = deque(maxlen=LOG_LEN)
col_indexes = ["temperature", "el_loadprofile", "th_loadprofile",
               "heatpump", "bev_at_home", "bev_battery_soc",
               "bev", "pv", "current_demand", "current_production",
               "temp_residual", "current_residual", "home_battery_soc",
               "electrical_storage_charge", "electrical_storage_discharge",
               "done", "action", "reward", "bev_charge_flag",
               "heatpump_flag", "thermal_storage", "th_soc",
               "cost", "specific_cost"]


class ComplexEMS:
    def __init__(self, ep_len=96, log_file="results_log.csv"):
        super(ComplexEMS, self).__init__()
        # gym declarations
        self.observations = 9
        self.ep_len = ep_len
        self.timebase = 15/60

        # operational variables
        self.time = 0
        self.cost_per_kwh = {"regular": 0.3,
                             "PV": 0.08,
                             "Wind": 0.12,
                             "CHP": 0.15,
                             "El_Store": 0.1}

        # self.production_per_tech = {"regular": 0,
        #                             "PV": 0,
        #                             "Wind": 0,
        #                             "CHP": 0,
        #                             "El_Store": 0}

        self.tech = ["regular", "PV", "Wind", "CHP", "El_Store"]
        self.el_loadprofile = pd.read_csv("./Input_House/Base_Szenario/baseload_household.csv",
                                          delimiter="\t")["0"]/1000
        self.th_loadprofile = self.init_up()
        self.temperature = self.init_temperature()
        self.heatpump = self.init_heatpump(power=4)
        self.heatpump_flag = 0
        self.bev = self.init_bev(16)
        self.bev_battery_soc = 16
        self.bev_charge_flag = 0
        self.pv = self.init_pv()
        self.pv_multiplier = 16
        self.el_storage = self.init_el_storage()
        self.th_storage = self.init_th_storage()
        self.log = deque(maxlen=LOG_LEN)
        self.log_file = log_file
        self.reward_factor = 1
        self.max_cost = max(self.cost_per_kwh.values()) * 23 / 4
        self.action_space = ActionSpaceGenerator()

    def log_result(self, log):
        self.log.append(log)
        if len(self.log) == LOG_LEN:
            pd.DataFrame(self.log, columns=col_indexes).to_csv(self.log_file)
            self.log = deque(maxlen=LOG_LEN)

    def init_temperature(self):
        temperature = pd.read_csv('./Input_House/heatpump_model/mean_temp_hours_2017_indexed.csv',
                                  index_col="time")
        df = pd.DataFrame(index=pd.date_range("2017", periods=35040, freq='15min', name="time"))
        temperature.index = pd.to_datetime(temperature.index)
        df["quart_temp"] = temperature
        df.interpolate(inplace=True)
        return df

    def init_heatpump(self, power):
        start = '2017-01-01 00:00:00'
        end = '2017-12-31 23:45:00'
        rampUpTime = 1/15  # timesteps
        rampDownTime = 1/15  # timesteps
        minimumRunningTime = 1  # timesteps
        minimumStopTime = 2  # timesteps
        timebase = 15/60

        hp = heat_pump(identifier="hp_1",
                       timebase=timebase,
                       heatpump_type="Air",
                       heat_sys_temp=60,
                       environment=None,
                       userProfile=None,
                       rampDownTime=rampDownTime,
                       rampUpTime=rampUpTime,
                       minimumRunningTime=minimumRunningTime,
                       minimumStopTime=minimumStopTime,
                       heatpump_power=power,
                       full_load_hours=2100,
                       heat_demand_year=None,
                       building_type='DE_HEF33',
                       start=start,
                       end=end,
                       year='2017')

        hp.lastRampUp = self.time
        hp.lastRampDown = self.time
        return hp

    def init_bev(self, battery_max):
        start = '2017-01-01 00:00:00'
        end = '2017-12-31 23:45:00'
        bev = battery_electric_vehicle(timebase=15 / 60,
                                       identifier='bev_1',
                                       start=start,
                                       end=end,
                                       time_freq="15 min",
                                       battery_max=battery_max,
                                       battery_min=0,
                                       battery_usage=1,
                                       charging_power=11,
                                       chargeEfficiency=0.98,
                                       environment=None,
                                       userProfile=None)
        bev.prepareTimeSeries()
        return bev

    def init_pv(self):
        latitude = 50.941357
        longitude = 6.958307
        name = 'Cologne'

        weather_data = pd.read_csv("./Input_House/PV/2017_irradiation_15min.csv")
        weather_data.set_index("index", inplace=True)

        pv = photovoltaic(timebase=15,
                          identifier=name,
                          latitude=latitude,
                          longitude=longitude,
                          modules_per_string=5,
                          strings_per_inverter=1)
        pv.prepareTimeSeries(weather_data)
        pv.timeseries.fillna(0, inplace=True)
        return pv

    def init_el_storage(self):
        es = electrical_energy_storage(15, "el_store_1", 25, 0.9, 0.9, 5 * 15 / 60, 1)
        # es.prepareTimeSeries()
        return es

    def init_up(self):
        yearly_heat_demand = 2500  # kWh
        target_temperature = 60  # °C
        up = user_profile(heat_sys_temp=target_temperature,
                          yearly_heat_demand=yearly_heat_demand,
                          full_load_hours=2100)
        up.get_heat_demand()
        return up.heat_demand

    def init_th_storage(self):
        #Values for Thermal Storage
        target_temperature = 60  # °C
        hysteresis = 5  # °K
        mass_of_storage = 500  # kg
        timebase = 15

        ts = thermal_energy_storage(timebase,
                                    mass=mass_of_storage,
                                    hysteresis=hysteresis,
                                    target_temperature=target_temperature)#, userProfile=self.th_loadprofile)
        return ts

    # def calc_cost(self, tech):
    #     return self.cost_per_kwh[tech] * self.production_per_tech[tech], self.production_per_tech[tech]
    #
    # def calc_cost_current_mix(self):
    #     costs_production = np.array(list(map(self.calc_cost, self.tech)))
    #     cost_sum, production_sum = np.sum(costs_production, axis=0)
    #     return cost_sum/production_sum

    def set_action(self, action):
        return self.action_space.get_action(action)

    def calc_cost_current_mix(self, production_per_tech):
        cost, prod = 0, 0
        for tech in self.cost_per_kwh:
            cost += self.cost_per_kwh[tech] * production_per_tech[tech]
            prod += production_per_tech[tech]
        return cost/prod, cost

    def static_reset(self, time):
        self.time = time
        self.rand_start = time
        self.heatpump.lastRampUp = self.time
        self.heatpump.lastRampDown = self.time
        self.heatpump_flag = 0
        self.bev_charge_flag = 0
        self.th_storage.current_temperature = 60
        self.th_storage.state_of_charge = self.th_storage.mass * self.th_storage.cp * \
                                          (self.th_storage.current_temperature + 273.15)
        self.el_storage.stateOfCharge = 0
        self.bev_battery_soc = 16
        state = self.step(0)[0]  # np.array(np.zeros(self.observations))
        self.variables = {"heatpump_flag": self.heatpump_flag,
                          "store_temp": self.th_storage.current_temperature,
                          "el_store": self.el_storage.stateOfCharge,
                          "bev_battery_soc": self.bev_battery_soc,
                          "bev_charge_flag": self.bev_charge_flag,
                          "time": self.time}
        return state

    def reset(self):
        self.rand_start = int(np.random.rand()*(35040-self.ep_len))
        self.time = self.rand_start
        self.heatpump.lastRampUp = self.time
        self.heatpump.lastRampDown = self.time
        self.heatpump_flag = 0
        self.bev_charge_flag = 0
        self.th_storage.current_temperature = 60 - (np.random.random() - 0.5) * 2 * 4.5
        self.th_storage.state_of_charge = self.th_storage.mass * self.th_storage.cp * \
                                          (self.th_storage.current_temperature + 273.15)
        self.el_storage.stateOfCharge = np.random.random() if np.random.random() < 0.5 else 0
        self.bev_battery_soc = np.random.random()*16
        state = self.step(0)[0]  # np.array(np.zeros(self.observations))
        self.variables = {"heatpump_flag": self.heatpump_flag,
                          "store_temp": self.th_storage.current_temperature,
                          "el_store": self.el_storage.stateOfCharge,
                          "bev_battery_soc": self.bev_battery_soc,
                          "bev_charge_flag": self.bev_charge_flag,
                          "time": self.time}
        return state

    def step(self, action, EVALUATION=False):
        """
        demand - non controllable     = LoadProfile
        production - non controllable = Photovoltaic, Wind
        demand - controllable         = Heatpump, BEV
        production - controllable     = CombinedHeatAndPower

        cost per kWh:
        regular mix             = 0.25 €/kWh
        Photovoltaic            = 0.08 €/kWh
        Wind                    = 0.12 €/kWh
        CombinedHeatAndPower    = 0.15 €/kWh
        Electrical Storage      = 0.10 €/kWh

        possibilities:
        reward_1 = current mixed price <-- way to go
        reward_2 = mixed price per episode

        actions:
        0: Nothing
        1: Heatpump on/off
        2: BEV on

        3: CHP on/off
        (4: Electrical Storage (?) Wie modelliert man Netzverträglichkeit?
        Ansonsten macht es keinen Sinn Speicher bei Überschuss nicht zu laden!)

        states:
        (time?), current_demand, current_production, thermal_storage_temp, (electrical_storage?),
        current_residual, bev at home

        reward shaping:
        main component = negative price of current mix, normalized between -0.08 and -0.25
        th_storage = gets negative as temp drops/rises from hysteresis, normal = 0
        bev = gets negative if user leaves with less than 99% - 90% charge, normal = 0
        invalid actions = -10 e.g. turning off heatpump too early

        problems:
        chp + heatpump need to be specifically configured for heat demand
        how to determine deviation of forecast to not achieve perfect forecast but 80% or sth.
        NaN's in pv
        Forecasting over several techs, storage temp, residual, bev etc.
        forecasting temp needs to be done iteratively
        forecasting bev and residual should be deterministic
        Is search depth limitation necessary?

        simplifications for first build:
        dismiss wind and chp

        todo:
        calculate min/max of forecast
        calculate sizes of all techs
        Implement BEV reward --> not fully loaded etc

        :return:
        """
        # step 0: apply actions
        heatpump_action, bad_action = 0, False
        (hp_scale, bev_scale, store_scale), (self.heatpump_flag, self.bev_charge_flag, el_charge) = self.set_action(action)

        # step 1: calculate all demands and productions
        temperature = self.temperature.iat[self.time, 0]
        el_loadprofile = self.el_loadprofile.iat[self.time]
        th_loadprofile = self.th_loadprofile.iat[self.time, 0]
        heatpump = self.heatpump.heatpump_power * self.heatpump_flag * self.timebase
        bev_at_home = self.bev.at_home.iat[self.time, 0]
        self.bev_battery_soc, bev, self.bev_charge_flag = self.bev.charge_timestep(bev_at_home,
                                                                                   self.bev_charge_flag,
                                                                                   self.bev_battery_soc)
        bev *= bev_scale
        heatpump *= hp_scale
        pv = self.pv.timeseries.iat[self.time, 0] * self.pv_multiplier
        current_demand = el_loadprofile + heatpump + bev
        current_production = pv
        temp_residual = current_demand - current_production

        # step 2. calculate storage states
        electrical_storage_charge, electrical_storage_discharge = 0, 0
        thermal_storage, th_soc, bad_action = self.operate_th_storage(th_loadprofile,
                                                                      hp_scale,
                                                                      temperature)

        if el_charge:
            bad_action = True

        if temp_residual < 0 \
                and (self.el_storage.stateOfCharge/self.el_storage.capacity) != 1 \
                and el_charge:
            electrical_storage_charge = np.clip(abs(temp_residual), 0, self.el_storage.maxPower) * store_scale
            self.el_storage.charge(electrical_storage_charge, 15, self.time)
            bad_action = False

        if temp_residual > 0 \
                and (self.el_storage.stateOfCharge/self.el_storage.capacity) != 0 \
                and el_charge:
            electrical_storage_discharge = np.clip(abs(temp_residual), 0, self.el_storage.maxPower) * store_scale
            self.el_storage.discharge(electrical_storage_discharge, 15, self.time)
            bad_action = False

        # step 3: calculate residual load
        current_demand += electrical_storage_charge
        current_production += electrical_storage_discharge
        current_residual = current_demand - current_production

        # step 4: calculate validity and reward
        production_per_tech = {"regular": current_residual,
                                    "PV": pv,
                                    "Wind": 0,
                                    "CHP": 0,
                                    "El_Store": electrical_storage_discharge}
        if current_residual < 0:
            production_per_tech = {"regular": 0,
                                        "PV": pv,
                                        "Wind": 0,
                                        "CHP": 0,
                                        "El_Store": electrical_storage_discharge}

        rew_factor = 2
        if current_residual < 0 and current_production > 0:
            rew_factor = ((current_production + current_residual) / current_production) * rew_factor

        specific_cost, full_cost = self.calc_cost_current_mix(production_per_tech)
        reward = (1-np.clip(specific_cost * 4, 0, 1)) * rew_factor + 0.1
        # reward = 1 - (specific_cost - min(self.cost_per_kwh.values())) / \
        #          (max(self.cost_per_kwh.values()) - min(self.cost_per_kwh.values())) * rew_factor
        # reward = 1 - (production_per_tech["regular"] * self.cost_per_kwh["regular"]) / self.max_cost

        if current_residual == 0:
            reward += 0.3

        if thermal_storage < 55 or thermal_storage > 65:
            bad_action = True

        if self.bev.at_home.iat[self.time-1, 0] == 1 and bev_at_home == 0:
            if self.bev_battery_soc <= self.bev.battery_max * 0.95:
                bad_action = True

        if bev_at_home == 0 and self.bev_charge_flag:
            bad_action = True

        if self.bev_charge_flag and self.bev_battery_soc == self.bev.battery_max:
            bad_action = True

        # step 5: calculate states
        # normalizing factors:
        # maximum expected residual load = heatpump + bev + el_loadprofile = 2kW + 11kW + ~10kW = 23 kW
        # th_storage temp = 55 - 65 °C
        done = self.time >= self.rand_start + self.ep_len

        state = np.array([(thermal_storage-55) / 10,
                          self.bev_battery_soc / self.bev.battery_max,
                          self.el_storage.stateOfCharge / self.el_storage.capacity])

        if bad_action:
            reward = -1
            done = True
        self.time += 1
        self.variables = {"heatpump_flag": self.heatpump_flag,
                          "store_temp": th_soc,
                          "el_store": self.el_storage.stateOfCharge,
                          "bev_battery_soc": self.bev_battery_soc,
                          "bev_charge_flag": self.bev_charge_flag,
                          "time": self.time}

        if EVALUATION:
            log = [temperature, el_loadprofile, th_loadprofile, heatpump, bev_at_home, self.bev_battery_soc, bev, pv, current_demand,
                   current_production, temp_residual, current_residual, self.el_storage.stateOfCharge, electrical_storage_charge,
                   electrical_storage_discharge, done, action, reward, self.bev_charge_flag, self.heatpump_flag,
                   thermal_storage, th_soc, full_cost, specific_cost]
            self.log_result(log)

        return state, reward/self.reward_factor, done, self.variables

    def operate_th_storage(self, heat_demand, hp_action, temperature):
        feedback = True
        # if hp_action:
        #     if self.heatpump_flag:
        #         feedback = self.heatpump.rampUp(self.time)
        #     else:
        #         feedback = self.heatpump.rampDown(self.time)
        #     if feedback is None:
        #         feedback = True
        if not hp_action==0:
            heat_production = self.heatpump.heatpump_power * self.heatpump.get_current_cop(temperature) * hp_action
        else:
            heat_production = 0
        temp, soc = self.th_storage.operate_storage_reinforcement(heat_demand, heat_production)
        return temp, soc, not feedback
