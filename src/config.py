missing_maps = {

    "Tenure": "mean",
    "PreferredLoginDevice": "mode",
    "CityTier": "mode",
    "WarehouseToHome": "mean",
    "PreferredPaymentMode": "mode",
    "HourSpendOnApp":"mean",
    "NumberOfDeviceRegistered": "mean",
    "PreferedOrderCat": "mode",
    "SatisfactionScore":"mean",
    "NumberOfAddress":"mean",
    "Complain":"mode",
    "OrderAmountHikeFromlastYear":"mean",
    "CouponUsed": "mean",
    "OrderCount":"mean",
    "DaySinceLastOrder":"mean",
    "CashbackAmount":"mean"
}


value_maps = {
    "PreferredLoginDevice": {
        "Mobile Phone": "Mobile Phone",
        "Phone": "Mobile Phone",
        "Computer": "Computer"
    },
    "PreferredPaymentMode": {
        "Cash on Delivery": "Cash on Delivery",
        "COD": "Cash on Delivery",
        "CC": "Credit Card",
        "Credit Card": "Credit Card",
        "E-Wallet": "E-Wallet",
        "UPI": "E-Wallet",
        "Debit Card": "Debit Card"
    },
    "PreferedOrderCat": {
    "Mobile Phone": "Mobile Phone",
    "Phone": "Mobile Phone",
    "Laptop & Accessory": "Laptop & Accessory",
    "Fashion": "Fashion",
    "Grocery": "Grocery",
    "Others" : "Others"
    }
}


features  = { "Tenure", "Complain", "CashbackAmount", "SatisfactionScore", "DaySinceLastOrder", 
                              "WarehouseToHome", "OrderAmountHikeFromlastYear", "NumberOfAddress"}