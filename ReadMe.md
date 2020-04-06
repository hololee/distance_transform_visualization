<h2>Distance Transform by multiple distance method</h2>

Distance transform is two pass algorithm. using local mask.
  
In first pass, calculate weight of distance at left, top part of filer.  
In second pass, calculate weight of distance at right, bottom part of filter.  
  
*Below figure is simple random points distance transform result using this code.*  
  
---
* Euclidean 3 by 3 neighborhood calculation.(main.py)  
![Euclidean 3by3 neighborhood calculation](./D_e_3.png)

---
* Euclidean 5 by 5 neighborhood calculation.(main2.py)  
![Euclidean 3by3 neighborhood calculation](./D_e_5.png)

---
* Euclidean 7 by 7 neighborhood calculation.(main3.py)  
![Euclidean 3by3 neighborhood calculation](./D_e_7.png)