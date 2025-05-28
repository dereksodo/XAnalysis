from sklearn.metrics import make_scorer
from src.DataProcessing.evaluate import compute_metrics
import numpy as np
import os
class NamesAndCodes:
    country_codes = [
        "CAN", "USA", "GBR", "FRA", "DEU", "ITA", "JPN",        # G7
        "BRA", "RUS", "IND", "CHN", "ZAF",                      # BRICS
        "MEX", "ARG", "TUR", "KOR", "IDN", "AUS", "SAU",        # Seven key economies
        "EUU"                                                   # European Union
    ]
    feature_codes = [["NV.AGR.TOTL.ZS","SP.RUR.TOTL.ZS"], # agriculture-and-rural-development
                     ["DT.ODA.ODAT.CD","DT.ODA.ODAT.GN.ZS"], # aid-effectiveness
                     ["EN.ATM.GHGT.KT.CE","EG.FEC.RNEW.ZS"], # climate-change
                     ["NY.GDP.MKTP.CD","NY.GDP.MKTP.KD.ZG"], # economy-and-growth
                     ["SE.TER.ENRR","SE.XPD.TOTL.GD.ZS"], # education
                     ["EG.USE.PCAP.KG.OE","EG.ELC.ACCS.ZS"], # engergy-and-mining
                     ["EN.ATM.PM25.MC.ZS","SH.H2O.SMDW.ZS"], # environment
                     ["DT.DOD.DECT.CD","DT.DOD.DECT.GN.ZS"], # external-debt
                     ["FS.AST.PRVT.GD.ZS","FX.OWN.TOTL.ZS"], # financial-sector
                     ["SG.GEN.PARL.ZS","SL.TLF.TOTL.FE.ZS"], # gender
                     ["SP.DYN.LE00.IN","SH.DYN.MORT"], # health
                     ["IT.NET.USER.ZS","IT.CEL.SETS.P2"], # infrastructure
                     ["SI.POV.NAHC","SI.POV.GINI"], # poverty
                     ["IC.BUS.NREG","IC.BUS.EASE.XQ"], # private-sector
                     ["GC.XPN.TOTL.GD.ZS","GC.TAX.TOTL.GD.ZS"], # public-sector
                     ["GB.XPD.RSDV.GD.ZS","TX.VAL.TECH.MF.ZS"], # science-and-technology
                     ["SP.DYN.LE00.MA.IN","SL.UEM.TOTL.MA.ZS"], # social-development
                     ["per_allsp.cov_pop_tot","per_allsp.adq_pop_tot"], # social-protection-and-labor 
                     ["NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS"], # trade
                     ["SP.URB.TOTL.IN.ZS","EN.POP.SLUM.UR.ZS"] # urban-development
    ]

class selected_features:
    SF = [  "SP.DYN.LE00.IN",
            "SP.URB.TOTL.IN.ZS",
            "NV.AGR.TOTL.ZS",
            "EG.USE.PCAP.KG.OE",
            "FS.AST.PRVT.GD.ZS",
            "NE.IMP.GNFS.ZS",
            "NY.GDP.MKTP.CD",
            "NE.EXP.GNFS.ZS",
            "NY.GDP.MKTP.KD.ZG",
            "EN.ATM.GHGT.KT.CE"
    ]

class get_scores:
    @staticmethod
    def feasibility_score(metrics):
        score = 0
        score += int(metrics["rmse/std"] < 1)
        score += int(metrics["r2"] > 0.6)
        score += int(metrics["mase"] < 1)
        score += int(metrics["da"] > 0.7)
        return score

    @staticmethod
    def guiding_score(metrics):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        rmse_std = metrics["rmse/std"]
        r2 = metrics["r2"]
        mase = metrics["mase"]
        da = metrics["da"]
        score = (
            5 * sigmoid((da - 0.7) / 0.7) +
            2 * sigmoid((r2 - 0.6) / 0.6) +
            2 * sigmoid(-(rmse_std - 1)) +
            sigmoid(-(mase - 1))
        )
        return score

    @staticmethod
    def guiding_score_sklearn(estimator, X, y):
        y_pred = estimator.predict(X)
        metrics = compute_metrics(y, y_pred)
        return get_scores.guiding_score(metrics)

    guiding_score_scorer = make_scorer(guiding_score_sklearn, greater_is_better=True)
    
class FeatureNames:
    code2name = {
        "SP.DYN.LE00.IN": "Life Expectancy",
        "SP.URB.TOTL.IN.ZS": "Urban Population %",
        "NV.AGR.TOTL.ZS": "Agriculture GDP %",
        "EG.USE.PCAP.KG.OE": "Energy Use",
        "FS.AST.PRVT.GD.ZS": "Private Sector Assets %",
        "NE.IMP.GNFS.ZS": "Imports %",
        "NY.GDP.MKTP.CD": "GDP (USD)",
        "NE.EXP.GNFS.ZS": "Exports %",
        "NY.GDP.MKTP.KD.ZG": "GDP Growth %",
        "EN.ATM.GHGT.KT.CE": "GHG Emissions"
    }

class Paths:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    FIGURE_DIR = os.path.join(BASE_DIR, "figures")
    SRC_DIR = os.path.join(BASE_DIR, "src")