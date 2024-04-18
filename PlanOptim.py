import pandas as pd 
import numpy as np 
from inputs import *
from pulp import * 
from icecream as ic

class OPTIM_SD:
    def __init__(self, cell_info, demand):
        self.current_month = current_month
        self.Fire_decreas = Fire_decreas
        self.min_inventory = min_inventory
        self.max_inventory = max_inventory
        self.last_inven = last_inven
        self.fire_cost = fire_cost 
        
        self.t = list(range(elf.current_month, 13))
        self.Demand =(demand['SALES'] + demand['WF6']).to_dict()
        self.GS_Unit_Product = (cell_info['AMP']/3500).to_dict()
        self.Operation_Max_Time = (cell_info['MAX_OPER']).to_dict()
        self.Operation_Min_Time = (cell_info['MIN_OPER']).to_dict()
        self.Total_Time = (cell_info['TOTAL_TIME']).to_dict()
        self.Power_Unit_Usage = (cell_info['AMP']*cell_info['VOLT']/cell_info['EFFICIENCY']*100/1000).to_dict()
        self.Power_Unit_Price = (cell_info['POWER_PRICE']).to_dict()
        self.Fire_Unit_Price = (cell_info['FIRE']*self.fire_cost/12*(1-self.Fire_decrease)).to_dict()
        self.Weights = (cell_info['WEIGHTS']).to_dict()
        self.ind = cell_info.index.tolist()
        self.ind1, self.ind2, self.ind3 = zip(*self.ind)
        
    def MAX_INVN(self):
        # Setting the Problem 
        prob_invn = LpProblem("Production Planning 1", LpMaxmize)
        
        # Decision Variables 
        Operation_Time_t = LpVariable.dicts("Operation Time", [(i,j,k) for i,j,k in self.ind], 0)
        Production_t = LpVariable.dicts("Quantity Produced", self.t, 0)
        Inventory_t = LpVariable.dicts("Inventory", self.t, lowBound = self.min_inventory, upBound=self.max_inventory)
        
        # Objective Function 
        prob_invn += lpSum(Inventory_t[i] for i in self.t)
        
        # Constraints 
        for k in self.t[1:]:
            prob_invn += Production_t[k]*1000 - sum(
                self.GS_Unit_Product[i,j,k] * Operation_Time_t[i,j,k] for in zip(self.ind1[::(len(self.t))-1], self.ind2[::(len(self.t))-1])
            ) ==0
        for i, j, k in self.ind:
            prob_invn += self.Operation_Min_Time[i,j,k] <= Operation_Time_t[i,j,k] <= self.Operation_Max_Time[i,j,k]
            
        Inventory_t[self.current_month] = self.last_inven
        for i in self.t[1:]:
            prob_invn += (Production_t[i] + Inventory_t[i-1] - Inventory_t[i]) == self.Demand[i] 

        prob_invn.solve()
        print("Step1 Solution Status =", LpStatus[prob_invn.status])
        
        # Product Output 
        prod_arr = []
        for v in prob_invn.variables():
            if v.name.startswith("Quantity_Produced"): 
                prod_row = np.narray([v.name.split('_')[-1], v.varValue])
                prod_arr.append(prod_row)
        prod_df = (
            pd.DataFrame(prod_arr, columns =['MONTH', 'PROD'])
            .astype({'MONTH':'int', 'PROD': 'float'})
            .sort_values('MONTH')
        )
        return prod_df
        
    def MIN_COST(self, prod_df, type, err=10):
        # Setting the Problem 
        prob = LpProblem("Production Planning 2", LpMinmize)
        # parameter 
        Product = prod_df.set_index('MONTH').to_dict()['PROD']
        
        # Decision Variables 
        Operation_Time_t = LpVariable.dicts("Operation Time", [(i,j,k) for i,j,k in self.ind], 0)
        Inventory_t = LpVariable.dicts("Inventory", self.t, 0)
        
        # Object Function 
        if type =='COST':
            prob += lpSum(
                self.Power_Unit_Usage[i,j,k] * Operation_Time_t[i,j,k] * self.Power_Unit_Price[i,j,k] for i,j,k in self.ind
            )
        elif type == 'FIRE':
            prob = lpSum(
                self.Fire_Unit_Price[i,j,k] * Operation_Time-t[i,j,k] / self.Total_Time[i,j,k] for i,j,k in self.ind
            )
        elif type =='COMPLEX':
            prob += lpSum(
                Operation_Time_t[i,j,k]*(
                    (self.Power_Unit_Usage[i,j,k] * self.Power_Unit_Price[i,j,k]) + (self.Fire_Unit_Price[i,j,k] / self.Total_Time[i,j,k])
                ) for i,j,k in self.ind
            )
        # Constraints
        for k in self.t[1:] : 
            prob += (
                Product[k]*1000 - sum(
                    self.GS_Unit_Product[i,j,k] * Operation_Time_t[i,j,k] for i,j in zip(self.ind1[::(len(self.t))-1], self.ind2[::(len(self.t))-1])
                ) >= -err/2
            ) and (
                Product[k]*1000 - sum(
                    self.GS_Unit_Product[i,j,k] * Operation_Time_t[i,j,k] for i,j in zip(self.ind1[::(len(self.t))-1], self.ind2[::(len(self.t))-1])
                ) <= err/2
            )
        
        for i,j,k in self.ind:
            prob += self.Operation_Min_Time[i,j,k] <= Operation_Time_t[i,j,k] <= self.Operation_Max_Time[i,j,k]

        Inventory_t[self.current_month] = self.last_invn
        
        for i in self.t[1:]:
            prob += (Product[i] + Inventory_t[i-1] - Inventory_t[i]) == self.Demnad[i] 
                
        prob.solve()
        print("Step 2 Solution Status = ", LpStatus[prob.status])
        
        # Print the solution of the Decision Variables 
        req_df = pd.DataFrame(self.Demand.items(), columns = ['MONTH', 'DEMAND'])
        oper_arr, invn_arr = [], []
        char_remov = ["(", ")", "'", ","]
        for v in prob.variables():
            if v.name.startswith("Operation Time") : 
                name = v.name.replace("Operation Time_", "")
                for char in char_remov:
                    name = name.replace(char, '')
                name_split = name.split("_")
                row = np.array([name[:3].replace("_", "-"), name_split[-2], name_split[-1], v.varValue])
                oper_arr.append(row)
            elif v.name.startswith("Inventory"):
                row = np.array([v.name.split("_")[-1], v.varValue])
                invn_arr.append(row)
                
        oper_df = (
            pd.DataFrame(oper_arr, columns = ['FACTORY', 'CELL', 'MONTH', 'OPER'])
            .astype({'MONTH':'int'})
            .set_index(['FACTORY', 'CELL', 'MONTH'])
        )
        invn_df = (
            pd.DataFrame(invn_arr, columns = ['MONTH', 'INVEN'])
            .astype({'MONTH':'int'})
            .sort_values('MONTH')
            .reset_index(drop=True)
        )
        sales_df = (
            req_df
            .merge(prod_df, on ='MONTH')
            .MERGE(invn_df, on ='MONTH')
        )

        print(f" \n{type} 판매량 & 생산량 & 재고량 \n")
        print(sales_df)

        return oper_df



                               
            


