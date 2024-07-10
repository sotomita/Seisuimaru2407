#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# module import
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm
import metpy.calc as mpcalc
from metpy.units import units


raw_columns = [
    "OBS_Time",
    "DCnt",
    "ST",
    "RE",
    "SondeN",
    "FCnt",
    "AGC",
    "rcvFREQ",
    "WM",
    "WD",
    "WS",
    "Height",
    "Xdistanc",
    "Ydistanc",
    "GF",
    "HDP",
    "PDP",
    "GeodetLat",
    "GeodetLon",
    "V",
    "Press0",
    "Temp0",
    "Humi0",
    "FE",
    "FRT",
    "FTI",
    "FVH",
    "FVL",
    "FSP1",
    "FSP2",
    "FSP3",
    "FSP4",
    "N",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "N8",
]  # noqa: E501

drop_columns = [
    "DCnt",
    "ST",
    "RE",
    "SondeN",
    "FCnt",
    "AGC",
    "rcvFREQ",
    "WM",
    "GF",
    "HDP",
    "PDP",
    "V",
    "FE",
    "FRT",
    "FTI",
    "FVH",
    "FVL",
    "FSP1",
    "FSP2",
    "FSP3",
    "FSP4",
    "N",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "N8",
]

new_columns = [
    "Time",
    "WD",
    "WS",
    "Height",
    "Dx",
    "Dy",
    "Lat",
    "Lon",
    "Pressure",
    "Temperature",
    "RH",
]


def data_qc(
    launch_time: datetime,
    sonde_no: str,
    raw_data_path: str,
) -> pd.DataFrame:
    """Read raw sounding data and convert them to analysis data files(.csv)
    Parameters
    ----------
    launch_time		:datetime
            launch time.
    sonde_no			:str
            sonde No.
    raw_data_path	:str
            path of raw data.
    Returns
    ----------
    df	:	pd.DataFrame
            analysis data file.
    """

    try:
        df = pd.read_csv(raw_data_path, skiprows=6)
        df.columns = raw_columns

        # Extract the target data
        df = df[df["SondeN"] == sonde_no]
        df = df.reset_index(drop=True)

        if len(df) == 0:
            df = None

        else:
            # Check if each record satisfies some conditions
            for i in tqdm(range(len(df))):
                """
                ST 観測ステータス
                                1:観測中／0: 放球前
                                2:受信機リモート／0:ローカル
                                4:BL点検済み／点検前
                                のOR論理演算値

                """
                st = df["ST"].loc[i]
                st = bin(st)

                """
                GPSフラグ
                                4: ディファレンシャル3次元測位
                                3: ディファレンシャル2次元測位
                                2: 単独3次元測位
                                1: 単独2次元測位
                                0:測位不能                
                """

                gf = df["GF"].loc[i]

                # 衛星数
                num_gps = df["N"].loc[i]

                # Remove "0:観測中" record.
                if st[-1] != "1":
                    df = df.drop(index=i, inplace=False)
                # Remove "BL点検前" record.
                elif st[-3] != "1":
                    df = df.drop(index=i, inplace=False)
                # Remove "0:測位不能" record.
                elif gf == 0:
                    df = df.drop(index=i, inplace=False)
                # Extract record that recive GPS data from 4 or more Sattellite
                elif num_gps < 4:
                    df = df.drop(index=i, inplace=False)

            # Drop some field
            df = df.drop(columns=drop_columns)
            df = df.reset_index(drop=True)
            df.columns = new_columns

            # Cast some fields
            cast_fields = [
                "WD",
                "WS",
                "Height",
                "Dx",
                "Dy",
                "Lat",
                "Lon",
                "Pressure",
                "Temperature",
                "RH",
            ]
            for cast_field in cast_fields:
                df[cast_field] = df[cast_field].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

            # Extract record if no field is NaN.
            df = df.dropna(how="any")
            df = df.reset_index(drop=True)

            # df is None if df has no records.
            if len(df) == 0:
                df = None
            else:

                # Calculate dewpoint temperature
                df["Dewpoint"] = np.nan
                for i in range(len(df)):
                    rh = df.loc[i, "RH"]
                    if rh <= 1e-4:
                        dewpoint = np.nan

                    else:
                        rh = 1.0 if rh > 100.0 else rh * 0.01
                        temp = df.loc[i, "Temperature"] * units.degC
                        dewpoint = mpcalc.dewpoint_from_relative_humidity(
                            temp,
                            rh,
                        ).m_as("degC")

                    df.loc[i, "Dewpoint"] = dewpoint

                # Set observation time of first record
                df_time0 = df.loc[0, "Time"]
                df_time0 = df_time0.split(":")
                df_time0 = [int(i) for i in df_time0]

                df.loc[0, "Time"] = datetime(
                    launch_time.year,
                    launch_time.month,
                    launch_time.day,
                    *df_time0,
                )

                # Set observation time of each record from the previous one
                for i in range(1, len(df)):
                    df_time = df.loc[i, "Time"]
                    df_time = df_time.split(":")
                    df_time = [int(i) for i in df_time]
                    df_time_previous = df.loc[i - 1, "Time"]

                    df.loc[i, "Time"] = datetime(
                        df_time_previous.year,
                        df_time_previous.month,
                        df_time_previous.day,
                        *df_time,
                    )

    except FileNotFoundError:
        df = None

    return df


def get_sonde_anl_data(
    sonde_dict: dict,
    raw_data_dir_path: str,
    anl_data_dir_path: str,
) -> bool:
    """Get analysis data files
    Parameters
    ----------
    sonde_dict	:str
            Its key means observation  Number, values is the list
              contains sonde No(:str) and launch time(:datetime).
            Example
                    sonde_dict = {"001":["0123456",datetime(2001,12,31,10,30)]}
    raw_data_dir_path	:str
            path of the raw data directory.
    anl_data_dir_path	:str
            path of the analysis data directory.
    Returns
    ----------
    bool
            True if successful,False otherwise.
    """
    for st_no in sonde_dict.keys():
        print("**************************************************")
        print(f"Station No. is\t{st_no}")
        sonde_no = sonde_dict[st_no][0]
        launch_time = sonde_dict[st_no][1]
        fpaths = glob(f"{raw_data_dir_path}/*{sonde_no}.CSV")
        if len(fpaths) == 0:
            print("No data file is found!")
            continue
        elif len(fpaths) > 1:
            print("Caution : multiple data files is found !")

        for i in range(len(fpaths)):
            fpath = fpaths[i]
            print(f"Read raw data file:\t{fpath}")
            df = data_qc(launch_time, sonde_no, fpath)

            if df is None:
                print(f"{sonde_no} Not Found")
                continue
            elif len(fpaths) == 1:
                anl_data_path = f"{anl_data_dir_path}/sonde_{st_no}.csv"
            else:
                anl_data_path = f"{anl_data_dir_path}/sonde_{st_no}_{i+1}.csv"
            print(f"Save data:\t{anl_data_path}")

            df.to_csv(anl_data_path)


if __name__ == "__main__":
    print("preprocessing.py")
    raw_data_dir_path = "data/raw"
    anl_data_dir_path = "data/anl"

    field_book_path = "data/field_book.csv"
    """
    field_book.csv
    StationN,SondeN,Launch_time_JST,Error
    001,01234567,2024-01-01_00:00,0
    (str,str,str,int)
    """

    field_book = pd.read_csv(
        field_book_path,
        dtype={
            "StationN": str,
            "SondeN": str,
            "Launch_time_JST": str,
            "Error": int,
        },
    )
    field_book["Launch_time_JST"] = pd.to_datetime(
        field_book["Launch_time_JST"], format="%Y-%m-%d_%H:%M"
    )
    field_book["StationN"] = field_book["StationN"].astype(str)
    field_book["SondeN"] = field_book["SondeN"].astype(str)

    sonde_dict = {
        field_book["StationN"].iloc[i]: [
            field_book["SondeN"].iloc[i],
            field_book["Launch_time_JST"].iloc[i],
        ]
        for i in range(len(field_book))
    }

    get_sonde_anl_data(
        sonde_dict,
        raw_data_dir_path,
        anl_data_dir_path,
    )
