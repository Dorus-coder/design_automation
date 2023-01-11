# Documentation design automation

## Goal

Create a Pareto optimal front with the lowest resistance and lowest hold capacity vs the shape with the highest resistance and highest hold capacity.

## Method
Try different configurations of parameters and reward the agent when the enviroment has a lower resistance. Multiple resistance calclations can be used simultaniously.


## Parameters

* offset control points
The locations of control points can be controlled be setting an offset from the  standard control points, which are at the corners of the boxed shape.

* wpa : waterplane area = L * B * C_w    
* c_B : L * b * T / displ

* lcb : longitudinal center of bouyancy forward of 0.5L in percentage of L

## problems to solve

1. I need an environment where coefficients, such as the block coefficient have a relationships with the main dimensions.
2. The height of the hold isn’t defined in Holtrop and Mennen. How do I deduce the draught and height of the hold?
3. What is the best way to define the draft?
4. Is there a relationship between lcb and cb, and if so, is the RL agent apable to detect it? There sure is one between the trim and the lcb, how can the agent detect those? Maybe it is better to approach the lcb mathematically.
5. How to define the holt and draft of the vessel?
6. ie bow angle

## solutions
 1. calculate the main dimension bases on the hold shape and add length fore and aft to the hold. with the block coefficient we can adjust the displacement or vice versa
 2. 
 3. We do not know the displacement exaclty, but we can limit the draught and optimize the vessel for a certain draft
 4. chapter 3.9.3 Swinging the curve in geometric properties of areas and volumes.
 5. No solution
 6. https://www.boatdesign.net/threads/definition-of-bow-entry-angle.61403/

## relationships

### displacement block coefficient and draft 
With the same displacement a higher block coefficient would decrease draft. or a higer draft with a higher block coefficient would increase the displacement.

## Best practice

* If a method returns a single parameter and takes no input I will use the property decorator.
* The B_spline should be incorperated in a parent class that could be inhereted by the classes that define the shape.
* seperate datclass for controlling the control points

## Input data

* minimum freeboard
the minimum freeboard is copied from ICLL 66 (International Convention Load Lines). The first column is the ships length, the seconth type "A" ships and the third type "B ships". Corrections on the standard freeboard are not made yet.
