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

class hyperparameters_tuning:
    para_tuning = {
        "XGBoost" : {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8},
        "RF" :      {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}
    }