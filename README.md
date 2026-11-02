# NetLogo_Bioscrape

> [!NOTE]
>This is an experimental binding between Bioscrape as an inference engine, and Netlogo as Frontend 
---

<details>
  <summary>🔍 first experiments screenshots </summary>
---

adaptiveness prototype
file source: Robustness_Simulator2, spatiotemporal recovery; 

https://github.com/user-attachments/assets/79c18d68-5479-485d-be40-a966c0c0138b

---
https://github.com/user-attachments/assets/ea0de096-7777-4899-895c-c4fb965382f8

Self stabilizing, colors indicate state red=stable, file source :Robustness_Simulator.nlogo

 this image shows NFBLB near adaptive but not robust enough against different param;  
 File source: folder / my adaptive model,
<img width="1196" height="517" alt="image" src="https://github.com/user-attachments/assets/d4506255-bce5-43f3-b0c5-4847291bebd8" />

This however shows the IFFLP that's adaptive and near perfect robustness. 
 <img width="1208" height="490" alt="image" src="https://github.com/user-attachments/assets/fea9ed7b-d02b-4fec-9f80-4f00f55a9f40" />

---
<img width="710" height="412" alt="image" src="https://github.com/user-attachments/assets/12aa5707-ebef-4e0a-ba96-933fc8655bf6" />

:white_check_mark: Done
:heavy_check_mark: reproduceable 


</details>


> [!IMPORTANT]
> Make sure to Inspect Netlogo codes and replace python environment and backend path according to your system.  


  # modeling with NFBLB & IFFLP motifs classes ; 

https://github.com/user-attachments/assets/c5328aaf-e112-4579-bb06-de226e7d0df8

| Color | Variable | Meaning                                   |
| ----- | -------- | ----------------------------------------- |
| Blue  | A        | Input-receiving component                 |
| Green | B        | Regulatory intermediate that inhibits C   |
| Red   | C        | Output used to evaluate adaptation        |


https://github.com/user-attachments/assets/67b62593-1b94-4130-a35b-7ce09f12a3db


> [!TIP]
> You can click the optimize button to recalculate projections and estimate better parameters
> You can also use the command line to perform thsee  actions; 


| Command              | Action                                                       |
| -------------------- | ------------------------------------------------------------ |
| optimize             | Start optimization; click/run again when ready to apply      |
| sensitivity-analysis | Run sensitivity analysis separately                          |
| robustness-report    | Request a local robustness report                            |
| show-parameters      | Print all 12 current parameters                              |
| save-data            | Save live data and parameter/stimulus events                 |
| save-results         | Save the latest completed analysis                           |
| inspect-data         | Open and inspect a saved JSON file                           |
| load-parameters      | Load a saved parameter configuration without resetting state |
