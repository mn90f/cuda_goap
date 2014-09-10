#pragma once

//Keep this in sync with kernel, be careful about changing the order of these things
enum EQUIPMENT{
	knife		= 1<<0,
	stone		= 1<<1,
	sword		= 1<<2,
	staff		= 1<<3,
	blade		= 1<<4,
	spear		= 1<<5,
	longSword	= 1<<6
};

const int NUMBER_OF_EQUIPMENT = 7;