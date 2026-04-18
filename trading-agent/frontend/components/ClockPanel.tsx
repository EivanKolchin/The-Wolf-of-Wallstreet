"use client";

import React, { useEffect, useState } from "react";

interface TimeZoneConfig {
    label: string;
    tz: string; 
}

export function ClockPanel() {
    const [time, setTime] = useState<Date | null>(null);
    const [zones, setZones] = useState<TimeZoneConfig[]>([]);

    useEffect(() => {
        // Initialize Default (Local time) + allow loading from settings
        const loadZones = () => {
            const saved = localStorage.getItem("user_timezones");
            if (saved) {
                try { 
                    const parsed = JSON.parse(saved);
                    setZones(parsed.length > 0 ? parsed : [{ label: "LOCAL", tz: "local" }]);
                } catch (e) {
                    setZones([{ label: "LOCAL", tz: "local" }]);
                }
            } else {
                setZones([{ label: "LOCAL", tz: "local" }]);
            }
        };

        loadZones();
        setTime(new Date());

        // Update time every second
        const timer = setInterval(() => setTime(new Date()), 1000);

        // Listen for storage changes from a hypothetical settings page
        window.addEventListener('storage', loadZones);
        
        return () => {
            clearInterval(timer);
            window.removeEventListener('storage', loadZones);
        };
    }, []);

    // Prevent hydration mismatch by returning empty structure matching the wrapper until mounted
    if (!time) return <div className="hidden lg:flex items-center space-x-6 border-r border-[#171717] pr-6 mr-1 h-8 min-w-[100px]"></div>;

    return (
        <div className="hidden lg:flex items-center space-x-6 border-r border-[#171717] pr-6 mr-1">
            {zones.map((zone, idx) => {
                const isLocal = zone.tz === "local";
                
                let timeString = "";
                let dateString = "";
                
                try {
                    timeString = isLocal 
                        ? time.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
                        : time.toLocaleTimeString('en-US', { timeZone: zone.tz, hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
                        
                    dateString = isLocal
                        ? time.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
                        : time.toLocaleDateString('en-US', { timeZone: zone.tz, weekday: 'short', month: 'short', day: 'numeric' });
                } catch (e) {
                    // Fallback in case of invalid timezone string in localStorage
                    timeString = time.toLocaleTimeString('en-US', { hour12: false });
                    dateString = time.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
                }

                return (
                    <div key={idx} className="flex flex-col items-end justify-center">
                        <div className="text-[10px] text-zinc-500 font-medium uppercase tracking-widest leading-none mb-1">
                            {zone.label} • {dateString}
                        </div>
                        <div className="text-[14px] text-zinc-200 font-mono leading-none tracking-wider">
                            {timeString}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
