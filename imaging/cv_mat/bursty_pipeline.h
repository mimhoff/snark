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

#pragma once

#include <tbb/parallel_pipeline.h>
#include <tbb/task_arena.h>
#include "../../tbb/bursty_reader.h"

namespace snark { namespace tbb {

/// run a tbb pipeline with bursty data using bursty_reader
/// @todo this class is too trivial; tear down?
template< typename T >
class bursty_pipeline
{
public:
    /// constructor
    /// @param number_of_threads maximum number of threads, 0 means auto
    bursty_pipeline( unsigned int number_of_threads = 0 );
    
    void run( bursty_reader< T >& reader, const ::tbb::filter< T, void >& filter );

private:
    unsigned int number_of_threads_;
};


template< typename T >
inline bursty_pipeline< T >::bursty_pipeline( unsigned int number_of_threads ) : number_of_threads_( number_of_threads > 0 ? number_of_threads : ::tbb::this_task_arena::max_concurrency() ) {}

template< typename T >
inline void bursty_pipeline< T >::run( bursty_reader< T >& reader, const ::tbb::filter< T, void >& filter )
{
    ::tbb::parallel_pipeline( number_of_threads_, reader.filter() & filter );
}

} } // namespace snark { namespace tbb {
