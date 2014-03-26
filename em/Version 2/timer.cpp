// FILE: timer.cpp

#include "timer.hpp"
#include <ctime>
#include <iostream>

void timer::start( void ) {
    timespec start;
    clock_gettime( CLOCK_REALTIME, &start );
    _ns_start = start.tv_nsec;
}

void timer::stop( void ) {
    timespec stop;
    clock_gettime( CLOCK_REALTIME, &stop );
    _ns_diff = stop.tv_nsec - _ns_start;
    if ( _ns_diff < 0 ) _ns_diff += 1000000000;
    ++_count;
    _sum_ns_diff += _ns_diff;
}

long timer::diffm( void ) const {
    return _ns_diff / 1000000;
}

long timer::diffu( void ) const {
    return _ns_diff / 1000;
}

long timer::diffn( void ) const {
    return _ns_diff;
}

void timer::printm( void ) const {
    std::cout << _name << ": " << ( _ns_diff / 1000000 ) << " ms" << std::endl;
}

void timer::printu( void ) const {
    std::cout << _name << ": " << ( _ns_diff / 1000 ) << " us" << std::endl;
}

void timer::printn( void ) const {
    std::cout << _name << ": " << _ns_diff << " ns" << std::endl;
}

long timer::adiffm( void ) const {
    return avg_ns_diff() / 1000000;
}

long timer::adiffu( void ) const {
    return avg_ns_diff() / 1000;
}

long timer::adiffn( void ) const {
    return avg_ns_diff();
}

void timer::aprintm( void ) const {
    std::cout << _name << ": " << ( avg_ns_diff() / 1000000 )
              << " ms Average" << std::endl;
}

void timer::aprintu( void ) const {
    std::cout << _name << ": " << ( avg_ns_diff() / 1000 )
              << " us Average" << std::endl;
}

void timer::aprintn( void ) const {
    std::cout << _name << ": " << avg_ns_diff()
              << " ns Average" << std::endl;
}

unsigned long timer::avg_ns_diff( void ) const {
    return _sum_ns_diff / _count;
}



