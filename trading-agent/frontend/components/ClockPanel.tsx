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
                    if (parsed.length > 0) {
                        setZones(parsed);
                        return;
                    }
                } catch (e) {
                    // fallthrough to default
                }
            }
            
            // Create default zones array with detected timezones
            let tzAbbr = "LOCAL";
            try {
                const parts = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' }).formatToParts(new Date());
                const tzPart = parts.find(p => p.type === 'timeZoneName');
                if (tzPart && tzPart.value) {
                    tzAbbr = `LOCAL (${tzPart.value})`;
                }
            } catch (e) {}

            const localTz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
            const defaultZones: TimeZoneConfig[] = [];

            // Add London if user is not in London
            if (!localTz.includes("Europe/London") && !localTz.includes("Europe/Belfast") && !localTz.includes("Europe/Dublin") && !localTz.includes("GB") && localTz !== "UTC") {
                defaultZones.push({ label: "LDN", tz: "Europe/London" });
            }
            
            // Add New York if user is not in New York
            if (!localTz.includes("America/New_York") && !localTz.includes("America/Detroit") && !localTz.includes("EST") && !localTz.includes("EDT")) {
                defaultZones.push({ label: "NY", tz: "America/New_York" });
            }
            
            // Add Local
            defaultZones.push({ label: tzAbbr, tz: "local" });
            
            setZones(defaultZones);
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
