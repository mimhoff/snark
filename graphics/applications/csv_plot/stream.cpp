// This file is part of snark, a generic and flexible library for robotics research
// Copyright (c) 2011 The University of Sydney
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University of Sydney nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
// IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/// @author Vsevolod Vlaskine

#include <boost/bind/bind.hpp>
#include "stream.h"
#include "traits.h"

namespace snark { namespace graphics { namespace plotting {

static const char* hex_color_( const std::string& c )
{
    if( c == "red" ) { return "#FF0000"; }
    if( c == "green" ) { return "#00FF00"; }
    if( c == "blue" ) { return "#0000FF"; }
    if( c == "yellow" ) { return "#FFFF00"; }
    if( c == "cyan" ) { return "#00FFFF"; }
    if( c == "magenta" ) { return "#FF00FF"; }
    if( c == "black" ) { return "#000000"; }
    if( c == "white" ) { return "#FFFFFF"; }
    if( c == "grey" ) { return "#888888"; }
    return &c[0];
}
    
stream::config_t::config_t( const comma::command_line_options& options )
    : csv( options )
    , size( options.value( "--size,-s,--tail", 10000 ) )
    , color_name( options.value< std::string >( "--color,--colour", "black" ) )
    , shape( options.value< std::string >( "--shape,--type", "curve" ) )
    , style( options.value< std::string >( "--style", "" ) )
    , weight( options.value( "--weight", 0.0 ) )
{
    if( csv.fields.empty() ) { csv.fields="x,y"; } // todo: parametrize on the graph type
    if( !color_name.empty() ) { color = QColor( hex_color_( color_name ) ); }
}

stream::stream( const stream::config_t& config )
    : config( config )
    , is_shutdown_( false )
    , is_stdin_( config.csv.filename == "-" )
    , is_( config.csv.filename, config.csv.binary() ? comma::io::mode::binary : comma::io::mode::ascii, comma::io::mode::non_blocking )
    , istream_( *is_, config.csv )
    , count_( 0 )
    , has_x_( config.csv.fields.empty() || config.csv.has_field( "x" ) )
    , buffers_( config.size )
{
}

stream::buffers_t_::buffers_t_( comma::uint32 size ) : x( size ), y( size ) {}

void stream::buffers_t_::add( const point& p )
{
    x.add( p.coordinates.x(), p.block );
    y.add( p.coordinates.y(), p.block );
}

bool stream::buffers_t_::changed() const { return x.changed(); }

void stream::buffers_t_::mark_seen()
{
    x.mark_seen();
    y.mark_seen();
}

void stream::start() { thread_.reset( new boost::thread( boost::bind( &graphics::plotting::stream::read_, boost::ref( *this ) ) ) ); }

bool stream::is_shutdown() const { return is_shutdown_; }

bool stream::is_stdin() const { return is_stdin_; }

void stream::shutdown()
{
    is_shutdown_ = true;
    if( thread_ ) { thread_->join(); }
    if( !is_stdin_ ) { is_.close(); }
}

void stream::read_()
{
    while( !is_shutdown_ && ( istream_.ready() || ( is_->good() && !is_->eof() ) ) )
    {
        const point* p = istream_.read();
        if( !p ) { break; }
        point q = *p;
        if( !has_x_ ) { q.coordinates.x() = count_; }
        ++count_;
        comma::synchronized< points_t >::scoped_transaction( points )->push_back( q );
    }
    is_shutdown_ = true;
}

bool stream::update()
{
    points_t p;
    {
        comma::synchronized< points_t >::scoped_transaction t( points );
        p = *t; // quick and dirty, watch performance?
        t->clear();
    }
    if( p.empty() ) { return false; }
    for( std::size_t i = 0; i < p.size(); ++i ) { buffers_.add( p[i] ); }
    bool changed = buffers_.changed();
    if( buffers_.changed() ) { update_plot_data(); }
    buffers_.mark_seen();
    return changed;
}

} } } // namespace snark { namespace graphics { namespace plotting {
