import "./globals.css";
import { Inter } from "next/font/google";
import Navbar from "@/components/Navbar";
import { Providers } from "./providers";
import { SetupModal } from "@/components/SetupModal";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "AI Trading Agent dashboard",
  description: "Real-time AI crypto trading terminal",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#000000] text-neutral-200 min-h-screen antialiased`}>
        <Providers>
          <SetupModal />
          <div className="flex flex-col h-screen overflow-hidden">
            <Navbar />
            <main className="flex-1 overflow-auto p-4 md:p-6 lg:p-8 space-y-4">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
